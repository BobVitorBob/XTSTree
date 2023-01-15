import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime


def split_ts4test(current_ts, days):
    chunks = list()
    chunk_size = 96
    chunk_start_list = np.arange(0, current_ts.shape[0], (chunk_size * days + 1)).tolist()
    for idx, value in enumerate(chunk_start_list):
        if (idx + 1) == len(chunk_start_list):
            save_ts_window(current_ts[value:],
                           str(days) + "dias_" + current_ts.columns[3] + "_" + current_ts.iloc[
                               value, 1] + " _ " + current_ts.iloc[-1, 1])
        else:
            save_ts_window(current_ts[value:(chunk_start_list[idx + 1])],
                           str(days) + "dias_" + current_ts.columns[3] + "_" + current_ts.iloc[
                               value, 1] + " _ " + current_ts.iloc[chunk_start_list[idx + 1], 1])
    return chunks


def split_ts4test_year(current_ts, days):
    chunks = list()
    chunk_size = 96
    chunk_start_list = np.arange(0, current_ts.shape[0], (chunk_size * days + 1)).tolist()
    for idx, value in enumerate(chunk_start_list):
        if (idx + 1) != len(chunk_start_list):
            old_date = datetime.strptime(current_ts.iloc[value, 1], "%Y-%m-%d").date()
            new_date = old_date + relativedelta(years=1)
            pos_next_year = np.where([current_ts.date == datetime.strftime(new_date, "%Y-%m-%d")])[1][0]
            print(pos_next_year)

            save_ts_window(current_ts[value:(chunk_start_list[idx + 1]) - 1],
                           "/anual/anual_" + str(days) + "dias_" + current_ts.columns[3] + "_" + current_ts.iloc[
                               value, 1] + " _ " + current_ts.iloc[chunk_start_list[idx + 1], 1] + "_TRAIN")
            save_ts_window(current_ts.iloc[pos_next_year:pos_next_year + chunk_start_list[1] - 1],
                           "/anual/anual_" + str(days) + "dias_" + current_ts.columns[3] + "_" + current_ts.iloc[
                               value, 1] + " _ " + current_ts.iloc[chunk_start_list[idx + 1], 1] + "_TEST")
    return chunks


def save_ts_window(current_ts, file_name, PATH="datasets/"):
    # current_ts.to_csv(PATH + current_ts[columns[3]] + "_" + current_ts[columns[2]] + ".csv", header=columns)
    current_ts.to_csv(PATH + file_name + ".csv")
    return True


ts = pd.read_csv('datasets/export_automaticas_23025122_umidrelmed2m.csv')
print(ts.shape)
split_ts4test(ts, 60)
# split_ts4test_year(ts, 20)
