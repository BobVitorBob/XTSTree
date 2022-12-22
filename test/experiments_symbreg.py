from XTSTree.XTSTreePageHinkley import XTSTreePageHinkley
from plot import plot
import time
import pandas as pd

from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.ensemble import RandomForestRegressor
from pysr import *

import numpy as np
import os


import wandb


def mae(y, y_hat):
    return np.mean(np.abs(y - y_hat))


def mse(y, y_hat):
    return np.mean(np.square(y - y_hat))


def rmse(y, y_hat):
    return np.sqrt(np.mean(np.square(y - y_hat)))


def mape(y, y_hat):
    return np.mean(np.abs((y - y_hat) / y) * 100)

# Median Relative Absolute Error (MdRAE)
# If our model’s forecast equals to the benchmark’s forecast then the result is 1.
# If the benchmarks forecast are better than ours then the result will be above > 1.
# If ours is better than it’s below 1.
def mdrae(y, y_hat, bnchmrk):
    return np.median(np.abs(y - y_hat) / np.abs(y - bnchmrk))


# Geometric Mean Relative Absolute Error (GMRAE)
def gmrae(y, y_hat, bnchmrk):
    abs_scaled_errors = np.abs(y - y_hat) / np.abs(y - bnchmrk)
    return np.exp(np.mean(np.log(abs_scaled_errors)))


# Mean Absolute Scaled Error (MASE)
def mase(y, y_hat, y_train):
    naive_y_hat = y_train[:-1]
    naive_y = y_train[1:]
    mae_in_sample = np.mean(np.abs(naive_y - naive_y_hat))
    mae = np.mean(np.abs(y - y_hat))

    return mae / mae_in_sample
def evaluate_ts(current_ts, model):
    X = []
    y = []
    current_ts["hour"] = pd.factorize(current_ts["hour"].astype(str).str[:2])[0]
    current_ts["date"] = pd.factorize(current_ts["date"])[0]
    for i, row in current_ts.iterrows():
        X.append([row['date'], row['hour']])
        y.append(float(row[-1]))

    model.fit(X, y, variable_names=['date', 'hour'])

    yhat = model.predict(X)
    perf_mae = round(mae(y, yhat),2)
    perf_mse = round(mse(y, yhat),2)
    perf_rmse = round(rmse(y, yhat),2)
    perf_mape = round(mape(y, yhat),2)

    return model, perf_mae, perf_mse, perf_rmse, perf_mape

def evaluate_ts_lag(current_ts, model, lag):
    X = []
    y = []
    current_ts["hour"] = pd.factorize(current_ts["hour"].astype(str).str[:2])[0]
    current_ts["date"] = pd.factorize(current_ts["date"])[0]
    for i, row in current_ts.iterrows():
        X.append([row['date'], row['hour'], current_ts.iloc[i - lag][2]])
        y.append(float(row[-1]))

    model.fit(X, y, variable_names=['date', 'hour', 'lag'])
    arr = []
    arr = model.get_best()['lambda_format'](np.array(X))
    #plot(arr, sec_plots=[y], divisions=[i for i, x in enumerate(X) if x[0] == 0])

    yhat = model.predict(X)
    perf_mae = round(mae(y, yhat),2)
    perf_mse = round(mse(y, yhat),2)
    perf_rmse = round(rmse(y, yhat),2)
    perf_mape = round(mape(y, yhat),2)
    return model, perf_mae, perf_mse, perf_rmse, perf_mape

def listing_all_files(PATH):
    dir_path = PATH
    res = []
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            res.append(path)
    return res


def get_regressor(criteria, file, cut):
    return PySRRegressor(
        binary_operators=['+', '-', '*', '/', 'pow'],
        unary_operators=['neg', 'exp', 'abs', 'log', 'sqrt', 'sin', 'tan', 'sinh', 'sign'],
        niterations=200,
        populations=30,
        population_size=60,
        progress=False,
        model_selection=criteria,
        equation_file="symbreg_objects/" + criteria + "_" + file + "_" + str(cut) +".csv"
        )



window_size = 96
s = 3.2
min_std = 0
max_std = 15
std = min_std
adf = 0.05

dir_path = 'datasets/umidrelmed2m/20dias/'
list_files = listing_all_files(dir_path)

list_criteria = ["best", "accuracy"]

list_XTSTree = [
                XTSTreePageHinkley(stop_condition='adf', stop_val=0.05, min_dist=30)
               ]

experiment_log = list()
for file in list_files:
    for sep in list_XTSTree:
        #file = "20dias_umidrelmed2m_2015-12-01 _ 2015-12-21.csv"
        print(file)
        series = pd.read_csv(dir_path+file).dropna()
        plot(series.umidrelmed2m, save=True, show=False, img_name="images/"+file+".pdf")

        t = time.perf_counter()
        xtstree = sep.create_splits(series.umidrelmed2m.values)
        t_diff = time.perf_counter() - t
        cuts = xtstree.cut_points()
        plot(series.umidrelmed2m, divisions=cuts, title=f'Segments with {adf} (ADF)', save=True, show=False, img_name="images/"+file+"_splits.pdf")

        print(cuts)
        for criteria in list_criteria:

            ### WANDB
            run = wandb.init(project="XTSTree", entity="barbon", reinit=True)
            ###

            model, raw_MAE, raw_MSE, raw_RMSE, raw_MAPE = evaluate_ts(series, get_regressor(criteria, file, 0))

            experiment_log_cuts = [[0, raw_MAE, raw_MSE, raw_RMSE, raw_MAPE, model.get_best()['equation'], criteria]]
            for idx, cut in enumerate(cuts):
                #print(idx,len(cuts))
                if idx == 0:
                    model, perf_MAE, perf_MSE, perf_RMSE, perf_MAPE = evaluate_ts(series.iloc[0:cut, :].copy(),
                                                                                  get_regressor(criteria, file, cut)) #WARM START?????
                elif idx == (len(cuts)-1):
                    model, perf_MAE, perf_MSE, perf_RMSE, perf_MAPE = evaluate_ts(series.iloc[cut:, :].copy(),
                                                                                  get_regressor(criteria, file, cut))
                else:
                    model, perf_MAE, perf_MSE, perf_RMSE, perf_MAPE = evaluate_ts(series.iloc[cut:cuts[idx+1], :].copy(),
                                                                                  get_regressor(criteria, file, cut))
                experiment_log_cuts.append([cut, perf_MAE, perf_MSE, perf_RMSE, perf_MAPE, model.get_best()['equation'], criteria])
            #print(experiment_log_cuts)
            df_experiment_log_cuts = pd.DataFrame(experiment_log_cuts)
            #print(df_experiment_log_cuts.shape)

            df_experiment_log_cuts.columns = ["Start", "MAE", "MSE", "RMSE", "MAPE", "Equation", "Criteria"]
            df_experiment_log_cuts.to_csv("logs/"+criteria+"_"+file+"_cuts_log.csv")

            experiment_log.append([file,
                                   type(sep).__name__,
                                   len(cuts),
                                   t, #time cost
                                   criteria, #parsimonly?
                                   raw_MAE,
                                   raw_MSE,
                                   raw_RMSE,
                                   raw_MAPE,
                                   df_experiment_log_cuts.MAE.drop([0], axis=0).mean(),
                                   df_experiment_log_cuts.MSE.drop([0], axis=0).mean(),
                                   df_experiment_log_cuts.RMSE.drop([0], axis=0).mean(),
                                   df_experiment_log_cuts.MAPE.drop([0], axis=0).mean(),
                                   ])
            ### WANDB
            wandb.log({"file":file,
                       "XTSTree":type(sep).__name__,
                       "Cuts": len(cuts),
                       "Time": t, #time cost
                       "Criteria": criteria, #parsimonly?
                       "MAE":raw_MAE,
                       "MSE":raw_MSE,
                       "RMSE":raw_RMSE,
                       "MAPE":raw_MAPE,
                       "MAE_leaves":df_experiment_log_cuts.MAE.drop([0], axis=0).mean(),
                       "MSE_leaves":df_experiment_log_cuts.MSE.drop([0], axis=0).mean(),
                       "RMSE_leaves":df_experiment_log_cuts.RMSE.drop([0], axis=0).mean(),
                       "MAPE_leaves":df_experiment_log_cuts.MAPE.drop([0], axis=0).mean()
                       })
            run.finish()
            ### WANDB


df_experiment_log = pd.DataFrame(experiment_log)
df_experiment_log.columns = ["File", "XTSTree", "Cuts", "Time", "Criteria", "MAE", "MSE", "RMSE", "MAPE",
                             "Mean_Leaf_MAE", "Mean_Leaf_MSE", "Mean_Leaf_RMSE", "Mean_Leaf_MAPE"]
df_experiment_log.to_csv("experiment_log_20.csv")