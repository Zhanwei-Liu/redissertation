"""Functions for data collection and cleaning."""
import os
from typing import List, Dict

import click
import numpy as np
import pandas as pd


SPC_REPORT_RENAME_DICT = {'yr': 'observation_year', 'mo': 'observation_month',
               'dy': 'observation_date', 'tz': 'timezone',
               'st': 'observation_state', 'mag': 'f_or_ef_scale',
               'slat': 'starting_latitude_deg',
               'slon': 'starting_longitude_deg',
               'elat': 'ending_latitude_deg',
               'elon': 'ending_longitude_deg'}


def get_tornado_reports(url_or_path: str, date_and_time_columns: List[str]=['date', 'time'],
                        column_rename_and_keep_dict: Dict[str, str]=SPC_REPORT_RENAME_DICT) -> pd.DataFrame:
    """Get and preprocess SPC tornado reports data."""
    df = (pd.read_csv(url_or_path,
                      parse_dates={'observation_datetime': date_and_time_columns})
            .rename(columns=column_rename_and_keep_dict)
            .loc[:, list(column_rename_and_keep_dict.values()) + ['observation_datetime',]])
    return df


def drop_unknown_f_or_ef_scales(df: pd.DataFrame, f_or_ef_scale_column: str, unknown_scale_id: int=-9) -> pd.DataFrame:
    """Drop observations with unknown E/F Scale values."""
    return df.loc[df[f_or_ef_scale_column] != unknown_scale_id].reset_index(drop=True)


def filter_by_year(df: pd.DataFrame, datetime_column: str, earliest_year_for_obs: int) -> pd.DataFrame:
    """Filter tornado reports data based on some start year."""
    return df.loc[df[datetime_column].dt.year >= earliest_year_for_obs]


def find_valid_time(df: pd.DataFrame, observation_time_column: str,
                    valid_time_column: str='observation_valid_time') -> pd.DataFrame:
    """Match up observed tornadoes with a valid time (12z-12z).

    Create the valid times of the reports, e.g. the "day" to which
    they are associated. We consider the "valid time" to be the
    ending time of a 24 hour forecast period. For example, a tornado
    occurring at 2016-05-18 05:00Z would have a valid time of
    2016-05-18 12:00Z, and a tornado occurring at 2016-05-18 15:00Z
    would have a valid time of 2016-05-19 12:00Z.
    """
    valid_times = (df.observation_datetime + pd.Timedelta('6H')).dt.ceil('12H')
    valid_times = valid_times.where(valid_times.dt.hour != 0, valid_times + pd.Timedelta('12H'))
    df = df.assign(**{valid_time_column: valid_times})
    return df


def filter_non_central_time_reports(df: pd.DataFrame, timezone_column: str,
                                    timezone_to_keep: int=3) -> pd.DataFrame:
    """Filter out tornado reports that weren't logged in CST."""
    return df.loc[df[timezone_column] == timezone_to_keep]


@click.command()
@click.option('--save-path', default='./', help='Location to save tornado reports data as csv file.')
@click.option('--csv-save-name', default='cleaned_tornado_reports.csv',
              help='Name of the saved tornado reports csv file.')
@click.option('--spc-tor-url', default='https://www.spc.noaa.gov/wcm/data/1950-2018_actual_tornadoes.csv',
              help="URL of SPC's tornado reports data.")
def get_clean_and_save_tornado_reports(save_path, csv_save_name, spc_tor_url):
    df = get_tornado_reports(spc_tor_url)
    df = (df.pipe(drop_unknown_f_or_ef_scales, 'f_or_ef_scale')
            .pipe(filter_non_central_time_reports, 'timezone')
            .pipe(filter_by_year, 'observation_datetime', 1990)
            .pipe(find_valid_time, 'observation_datetime'))
    df.reset_index(drop=True).to_csv(os.path.join(save_path, csv_save_name), index=False)


if __name__ == '__main__':
    get_clean_and_save_tornado_reports()