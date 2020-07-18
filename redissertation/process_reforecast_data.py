"""A script + fns to download and process Reforecast V3 data."""
import os
import logging
import pathlib
from typing import Iterable, Dict
from tempfile import TemporaryDirectory

import s3fs
import click
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed


S3_BUCKET = "noaa-gefs-retrospective"

BASE_S3_PREFIX = "GEFSv12/reforecast"

DAYS_PREFIX = {"1-10": "Days:1-10", "10-16": "Days:10-16"}

COMMON_COLUMNS_TO_DROP = ["valid_time", "surface"]

logger = logging.getLogger(__name__)


def create_selection_dict(
    latitude_bounds: Iterable[float],
    longitude_bounds: Iterable[float],
    forecast_days_bounds: Iterable[float],
) -> Dict[str, slice]:
    """Generate parameters to slice an xarray Dataset.

    Parameters
    ----------
    latitude_bounds : Iterable[float]
        The minimum and maximum latitude bounds to select.
    longitude_bounds : Iterable[float]
        The minimum and maximum longitudes bounds to select.
    forecast_days_bounds : Iterable[float]
        The earliest and latest forecast days to select.

    Returns
    -------
    Dict[str, slice]
        A dictionary of slices to use on an xarray Dataset.
    """
    latitude_slice = slice(max(latitude_bounds), min(latitude_bounds))
    longitude_slice = slice(min(longitude_bounds), max(longitude_bounds))
    first_forecast_hour = pd.Timedelta(f"{min(forecast_days_bounds)} days")
    last_forecast_hour = pd.Timedelta(f"{max(forecast_days_bounds)} days")
    forecast_hour_slice = slice(first_forecast_hour, last_forecast_hour)
    selection_dict = dict(
        latitude=latitude_slice, longitude=longitude_slice, step=forecast_hour_slice
    )
    return selection_dict


def reduce_dataset(
    ds: xr.Dataset, func: str = "mean", reduce_dim: str = "step"
) -> xr.Dataset:
    """Helper function to reduce xarray Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        A GEFS reforecast dataset.
    func : str, optional
        The reduction function to use, by default 'mean'
    reduce_dim : str, optional
        The dimension to reduce over, by default 'step'

    Returns
    -------
    ds : xr.Dataset
        The reduced dataset.
    """
    ds = getattr(ds, func)(reduce_dim)
    return ds


def try_to_open_grib_file(path: str,) -> xr.Dataset:
    """Try a few different ways to open up a grib file.

    Parameters
    ----------
    path : str
        Path pointing to location of grib file

    Returns
    -------
    ds : xr.Dataset
        The xarray Dataset that contains information
        from the grib file.
    """
    try:
        ds = xr.open_dataset(path, engine="cfgrib")
    except Exception as e:
        try:
            import cfgrib

            ds = cfgrib.open_datasets(path)
            ds = xr.combine_by_coords(ds)
        except:
            logger.error(f"Oh no! There was a problem opening up {path}: {e}")
            return
    return ds


def download_and_process_grib(
    s3_prefix: str,
    latitude_bounds: Iterable[float],
    longitude_bounds: Iterable[float],
    forecast_days_bounds: Iterable[float],
    save_dir: str,
    pressure_levels: Iterable[float] = [950.0, 500.0],
) -> str:
    """Get a reforecast grib off S3, process, and save locally as netCDF file.

    Parameters
    ----------
    s3_prefix : str
        S3 key/prefix/whatever it's called of a single grib file.
    latitude_bounds : Iterable[float]
        An iterable that contains the latitude bounds, in degrees,
        between -90-90.
    longitude_bounds : Iterable[float]
        An iterable that contains the longitude bounds, in degrees,
        between 0-360.
    forecast_days_bounds : Iterable[float]
        An iterable that contains the first/last forecast days.
    save_dir : str
        Local directory to save resulting netCDF file.
    pressure_levels : Iterable[float]
        Pressure levels to extract from fields that have pressure levels.

    Returns
    -------
    saved_file_path : str
        The location of the saved file.
    """
    base_file_name = s3_prefix.split("/")[-1]
    saved_file_path = os.path.join(save_dir, f"{base_file_name.split('.')[0]}.nc")
    if pathlib.Path(saved_file_path).exists():
        return saved_file_path

    logger.info(f"Processing {s3_prefix}")
    selection_dict = create_selection_dict(
        latitude_bounds, longitude_bounds, forecast_days_bounds
    )

    fs = s3fs.S3FileSystem(anon=True)
    try:
        with TemporaryDirectory() as t:
            grib_file = os.path.join(t, base_file_name)
            with fs.open(s3_prefix, "rb") as f, open(grib_file, "wb") as f2:
                f2.write(f.read())
            ds = try_to_open_grib_file(grib_file)
            if ds is None:
                return
            if "isobaricInhPa" in ds.coords:
                pressure_level = [
                    p for p in pressure_levels if p in ds["isobaricInhPa"]
                ]
                selection_dict["isobaricInhPa"] = pressure_level
            ds = ds.sel(selection_dict)
            # NOTE: The longitude is originally between 0-360, but
            # for our purpose, we'll convert it to be between -180-180.
            ds["longitude"] = (
                ("longitude",),
                np.mod(ds["longitude"].values + 180.0, 360.0) - 180.0,
            )
            ds = ds.assign(time=ds.time + ds["step"].max())
            if "pcp" in base_file_name:
                ds = ds.sum("step")
            else:
                ds = ds.mean("step")
            # now, we need to reshape the data
            ds = ds.expand_dims("time", axis=0).expand_dims("number", axis=1)
            # set data vars to float32
            for v in ds.data_vars.keys():
                ds[v] = ds[v].astype(np.float32)
            ds = ds.drop(COMMON_COLUMNS_TO_DROP, errors="ignore")
            ds.to_netcdf(saved_file_path, compute=True)
    except Exception as e:
        logging.error(f"Oh no! There was an issue processing {grib_file}: {e}")
        return
    logging.info(f"All done with {saved_file_path}")
    return saved_file_path


@click.command()
@click.argument("start_date")
@click.argument("end_date")
@click.option(
    "-df",
    "--date-frequency",
    default=1,
    help="Date frequency between start_date and end_date",
)
@click.option(
    "-m",
    "--members",
    default=["c00",],
    help="Gridded fields to download.",
    multiple=True,
)
@click.option(
    "-v",
    "--var-names",
    default=["cape_sfc", "cin_sfc", "hlcy_hgt"],
    help="Gridded fields to download.",
    multiple=True,
)
@click.option(
    "-p",
    "--pressure-levels",
    default=[950.0, 500.0],
    multiple=True,
    help="Pressure levels to use, if some pressure field is used.",
)
@click.option(
    "--latitude-bounds",
    nargs=2,
    type=click.Tuple([float, float]),
    default=(22, 55),
    help="Bounds for latitude range to keep when processing data.",
)
@click.option(
    "--longitude-bounds",
    nargs=2,
    type=click.Tuple([float, float]),
    default=(230, 291),
    help="Bounds for longitude range to keep when processing data, assumes values between 0-360.",
)
@click.option(
    "--forecast-days-bounds",
    nargs=2,
    type=click.Tuple([float, float]),
    default=(5.5, 6.5),
    help="Bounds for forecast days, where something like 5.5 would be 5 days 12 hours.",
)
@click.option(
    "--local-save-dir",
    default="./reforecast_v3",
    help="Location to save processed data.",
)
@click.option(
    "--final-save-path",
    default="./combined_reforecast_data.nc",
    help="Saved name of the combined netCDF file.",
)
@click.option(
    "--n-jobs", default=28, help="Number of jobs to run in parallel.",
)
def get_and_process_reforecast_data(
    start_date,
    end_date,
    date_frequency,
    members,
    var_names,
    pressure_levels,
    latitude_bounds,
    longitude_bounds,
    forecast_days_bounds,
    local_save_dir,
    final_save_path,
    n_jobs,
):
    # let's do some quick checks here...
    if not all([min(latitude_bounds) > -90, max(latitude_bounds) < 90]):
        raise ValueError(
            f"Latitude bounds need to be within -90 and 90, got: {latitude_bounds}"
        )
    if not all([min(longitude_bounds) >= 0, max(longitude_bounds) < 360]):
        raise ValueError(
            f"Longitude bounds must be positive and between 0-360 got: {longitude_bounds}"
        )
    if not all([min(forecast_days_bounds) >= 0, max(forecast_days_bounds) <= 16]):
        raise ValueError(
            f"Forecast hour bounds must be between 0-16 days, got: {forecast_days_bounds}"
        )
    if max(forecast_days_bounds) < 10:
        days = DAYS_PREFIX["1-10"]
    else:
        days = DAYS_PREFIX["10-16"]
    # now, let's make sure the local save directory exists.
    pathlib.Path(local_save_dir).mkdir(parents=True, exist_ok=True)
    globbed_list = [
        f'{S3_BUCKET}/{BASE_S3_PREFIX}/{dt.strftime("%Y/%Y%m%d00")}/{member}/{days}/{var_name}_{dt.strftime("%Y%m%d00")}_{member}.grib2'
        for dt in pd.date_range(start_date, end_date, freq=f"{date_frequency}D")
        for var_name in var_names
        for member in members
    ]
    logger.info(f"Number of files to be downloaded and processed: {len(globbed_list)}")
    # TODO: Should this be async and not parallelized?
    _ = Parallel(n_jobs=n_jobs, verbose=25)(
        delayed(download_and_process_grib)(
            f,
            latitude_bounds,
            longitude_bounds,
            forecast_days_bounds,
            local_save_dir,
            pressure_levels,
        )
        for f in globbed_list
    )

    ds = xr.open_mfdataset(os.path.join(local_save_dir, "*.nc"), combine="by_coords")
    ds.to_netcdf(final_save_path, compute=True)
    logger.info("All done!")


if __name__ == "__main__":
    get_and_process_reforecast_data()
