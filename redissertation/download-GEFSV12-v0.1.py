from ftplib import FTP
import pandas as pd
import os
import time
from datetime import date, timedelta

# If you manually interrupt the download, please modify {break_file = path + "/" + the interruptefilenamed} before next download
break_file = None
flag = True
start_time = '20040101'
end_time = '20091231'


def print_log(message, logfilename):
    with open(logfilename, 'a') as f:
        f.write(time.strftime("[%Y-%m-%d %H:%M:%S] ") + message + '\n')
    print(message)


def download(start_time, end_time, logfilename, break_file, flag):
    # connect to host, default port
    ftp = FTP('ftp.emc.ncep.noaa.gov',user='', passwd='', timeout=1200)
    # user anonymous, passwd anonymous@
    ftp.login()                          
    # change into "GEFSv12/reanalysis/FV3_reanalysis" directory
    # ftp.cwd('GEFSv12/reanalysis/FV3_reanalysis')
    all_dates = [date(int(start_time[:4]), int(start_time[4:6]), int(start_time[6:])) + timedelta(days=d) for d in 
       range((date(int(end_time[:4]), int(end_time[4:6]), int(end_time[6:])) - 
       date(int(start_time[:4]), int(start_time[4:6]), int(start_time[6:]))).days+1)]
    for curr_date in all_dates:
        path = 'GEFSv12/reanalysis/FV3_reanalysis/%s/%02d/%02d'%(curr_date.year, curr_date.month, curr_date.day)
        if not os.path.exists(path):
            os.makedirs(path)
        inittime = [0, 6, 12, 18]
        filenames = ['gec00.t%02dz.pgrb2.%s%02d%02d.0p25.f000'%(t, curr_date.year, curr_date.month, curr_date.day) for t in inittime]
        for filename in filenames:
            # Delete files that were interrupted during the previous download
            if os.path.exists(path + "/" + filename) and break_file == path + "/" + filename: os.remove(path + "/" + filename)

            if os.path.exists(path + "/" + filename):
                pass                        
            else:
                try:
                    start = time.time()
                    with open(path + "/" + filename, 'wb') as f:
                        ftp.retrbinary('RETR %s'%(path + "/" + filename), f.write)
                    end = time.time()
                    message = path + "/" + filename + 'Download speed: %.3f MB/s'%(os.path.getsize(path + "/" + filename)/1e6/(end-start))
                    print_log(message, logfilename)
                except EOFError:
                    message = 'EOFError： FTP Server Disconnected' + path + "/" + filename
                    print_log(message, logfilename)
                    break_file = path + "/" + filename
                    flag = True
                    return break_file, flag
                except ConnectionResetError:
                    message = 'ConnectionResetError: The remote host forced an existing connection to close' + path + "/" + filename
                    print_log(message, logfilename)
                    break_file = path + "/" + filename
                    flag = True
                    return break_file, flag
    ftp.quit()
    break_file = None
    flag = False
    return break_file, flag

logfilename = 'download-GEFSv12%s.log'%time.strftime("%Y-%m-%d-%H-%M-%S")
message = 'Downloading ...'
print_log(message, logfilename)
while flag:
    break_file, flag = download(start_time, end_time, logfilename, break_file, flag)
    message = 'Reconnected ...'
    print_log(message, logfilename)


