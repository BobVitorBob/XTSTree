from XTSTree.XTSTreeKSWIN import XTSTreeKSWIN
from XTSTree.XTSTreePageHinkley import XTSTreePageHinkley
from XTSTree.XTSTreePeriodicCut import XTSTreePeriodicCut
from XTSTree.XTSTreeRandomCut import XTSTreeRandomCut
from plot import plot
import time
import pandas as pd

import sys
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

    return model, model.get_best()['lambda_format'](np.array(X)), perf_mae, perf_mse, perf_rmse, perf_mape

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


def get_regressor(criteria, file, cut, iterations, path):
    return PySRRegressor(
        binary_operators=['+', '-', '*', '/', 'pow'],
        unary_operators=['neg', 'exp', 'abs', 'log', 'sqrt', 'sin', 'tan', 'sinh', 'sign'],
        niterations=iterations,
        populations=30,
        population_size=60,
        progress=False,
        model_selection=criteria,
        equation_file=f'{path}symbreg_objects/{criteria}_{cut}_{file}',
        verbosity = 0,
        temp_equation_file=False
        )

#param_niterations = 50

#if len(sys.argv) == 0:
#    print("-"*60)
#    print("Parameters: ")
#    print("- 1) path (ex. test/)")
#    print("- 2) Number Iterations Sym. Regression (default 50/)")
#else:
#    param_path = sys.argv[1]
    #param_path = ""
#    param_niterations = sys.argv[0]


param_dataset = sys.argv[1] #numero de dias
wandb_project_name = sys.argv[2]
wandb_entity = sys.argv[3]


#remover depois
param_path = "./test/"
param_niterations = 50

window_size = 96
s = 3.2
min_std = 0
max_std = 15
std = min_std
adf = 0.05

dir_path = param_path+f'datasets/umidrelmed2m/{param_dataset}/'
list_files = listing_all_files(dir_path)

list_criteria = ["best", "accuracy"]

list_XTSTree = [
                ['PageHinkley_adf', XTSTreePageHinkley(stop_condition='adf', stop_val=0.05, min_dist=30)],
                ['PageHinkley_depth', XTSTreePageHinkley(stop_condition='depth', stop_val=3, min_dist=30)],
                ['RandomCut_adf', XTSTreeRandomCut(stop_condition='adf', stop_val=0.05, min_dist=30)],
                ['RandomCut_depth', XTSTreeRandomCut(stop_condition='depth', stop_val=3, min_dist=30)],
                ['PeriodicCut_adf', XTSTreePeriodicCut(stop_condition='adf', stop_val=0.05, min_dist=30)],
                ['PeriodicCut_depth', XTSTreePeriodicCut(stop_condition='depth', stop_val=3, min_dist=30)],
                ]

df_experiment_log = pd.DataFrame(columns=["File", "XTSTree", "Cuts", "Time", "Criteria", "NumIterations", "time raw", "time leaves",
                             "MAE", "MSE", "RMSE", "MAPE",
                             "MAE_diff", "MSE_diff", "RMSE_diff", "MAPE_diff",
                             "Mean_Leaf_MAE", "Mean_Leaf_MSE", "Mean_Leaf_RMSE", "Mean_Leaf_MAPE"])

df_experiment_log.to_csv(param_path+f"experiment_log_{param_dataset}.csv", index=False)
for rep in range(1, 5):
    for file in list_files:
        for sep in list_XTSTree:
            separator_name = sep[0]
            separator = sep[1]

            print(file)
            series = pd.read_csv(dir_path+file).dropna()
            plot(series.umidrelmed2m, save=True, show=False, img_name=param_path+"images/"+file+".pdf")

            t = time.perf_counter()
            xtstree = separator.create_splits(series.umidrelmed2m.values)
            t_diff = time.perf_counter() - t
            cuts = xtstree.cut_points()
            plot(series.umidrelmed2m, divisions=cuts, title=f'Segments with {adf} (ADF)', save=True, show=False,
                 img_name=param_path+"images/"+file+"_rep"+str(rep)+"_splits.pdf")

            print("Cuts:", cuts)
            if len(cuts) == 0:
                print('0 cortes')
                continue
            for criteria in list_criteria:

                ### WANDB
                run = wandb.init(project=wandb_project_name, entity=wandb_entity, reinit=True, name=file+"_"+separator_name+"_rep"+str(rep)+"_"+criteria)
                try:
                  print('Avaliando time series inteira')
                  t_raw = time.perf_counter()
                  model, yhat, raw_MAE, raw_MSE, raw_RMSE, raw_MAPE = evaluate_ts(series, get_regressor(criteria, file, 0, param_niterations, param_path))
                  t_raw_diff = time.perf_counter() - t_raw
                  print('Terminei a série inteira')
                  plot(series.umidrelmed2m, save=True, show=False,
                        img_name=param_path + "images/" + file + "_splits_"+criteria+"_rep"+str(rep)+"_reg.pdf", sec_plots=[yhat])

                  experiment_log_cuts = [[0, raw_MAE, raw_MSE, raw_RMSE, raw_MAPE, model.get_best()['equation'], criteria, param_niterations, t_raw_diff]]
                  plot_cuts = list()
                  print('Avaliando folhas')
                  for start, finish in zip([0, *cuts], [*cuts, len(series.umidrelmed2m.values)]):
                      #print(idx,len(cuts))
                      t_cut = time.perf_counter()
                      model, yhat, perf_MAE, perf_MSE, perf_RMSE, perf_MAPE = evaluate_ts(series.iloc[start:finish, :].copy(),
                                                                                        get_regressor(criteria, file, finish, param_niterations, param_path))
                      t_cut_diff = time.perf_counter() - t_cut
                      plot_cuts.append(yhat)
                      experiment_log_cuts.append([finish, perf_MAE, perf_MSE, perf_RMSE, perf_MAPE,
                                                  model.get_best()['equation'], criteria, param_niterations, t_cut_diff])
                  print('Terminei as folhas')

                  #print(len(plot_cuts))
                  #print(np.concatenate(plot_cuts).ravel().tolist())

                  plot(series.umidrelmed2m, divisions=cuts, title=f'Segments with {adf} (ADF)', save=True, show=False,
                            img_name=param_path + "images/" + file + "_splits_"+criteria+"_rep"+str(rep)+"_cuts_reg.pdf", sec_plots=[np.concatenate(plot_cuts).ravel().tolist()])

                  df_experiment_log_cuts = pd.DataFrame(experiment_log_cuts)
                  #print(df_experiment_log_cuts.shape)

                  df_experiment_log_cuts.columns = ["Start", "MAE", "MSE", "RMSE", "MAPE",
                                                    "Equation", "Criteria", "NumIterations", "Time"]
                  df_experiment_log_cuts.to_csv(param_path+"logs/"+criteria+"_"+file+"_rep"+str(rep)+"_cuts_log.csv")
                  # Atualiza o dataframe de log do experimento conforme executa para ter o arquivo caso o experimento quebre
                  df_experiment_log = pd.DataFrame([[
                    file,
                    separator_name,
                    len(cuts),
                    t, #time cost
                    criteria, #parsimonly?
                    param_niterations,
                    t_raw_diff,
                    df_experiment_log_cuts.Time.drop([0], axis=0).sum(), #sum of leaves time
                    raw_MAE,
                    raw_MSE,
                    raw_RMSE,
                    raw_MAPE,
                    raw_MAE - round(df_experiment_log_cuts.MAE.drop([0], axis=0).mean(),2),
                    raw_MSE - round(df_experiment_log_cuts.MSE.drop([0], axis=0).mean(),2),
                    raw_RMSE - round(df_experiment_log_cuts.RMSE.drop([0], axis=0).mean(),2),
                    raw_MAPE - round(df_experiment_log_cuts.MAPE.drop([0], axis=0).mean(),2),
                    round(df_experiment_log_cuts.MAE.drop([0], axis=0).mean(),2),
                    round(df_experiment_log_cuts.MSE.drop([0], axis=0).mean(),2),
                    round(df_experiment_log_cuts.RMSE.drop([0], axis=0).mean(),2),
                    round(df_experiment_log_cuts.MAPE.drop([0], axis=0).mean(),2),
                  ]])
                  df_experiment_log.to_csv(param_path+f"experiment_log_{param_dataset}.csv", header=None, mode='a', index=False)
                  print(f'{file}-{criteria}-{separator_name} salvo localmente, subindo para o wandb')
                  ### WANDB
                  wandb.log({"file":file,
                              "XTSTree":separator_name,
                              "Cuts": len(cuts),
                              "Time": t_diff, #time cost
                              "Criteria": criteria, #parsimonly?
                              "NumIterations": param_niterations,
                              "Time raw": t_raw_diff,
                              "Time Leaves": df_experiment_log_cuts.Time.drop([0], axis=0).sum(),
                              "MAE":raw_MAE,
                              "MSE":raw_MSE,
                              "RMSE":raw_RMSE,
                              "MAPE":raw_MAPE,
                              "MAE_diff":raw_MAE - round(df_experiment_log_cuts.MAE.drop([0], axis=0).mean(),2),
                              "MSE_diff":raw_MSE - round(df_experiment_log_cuts.MSE.drop([0], axis=0).mean(),2),
                              "RMSE_diff":raw_RMSE - round(df_experiment_log_cuts.RMSE.drop([0], axis=0).mean(),2),
                              "MAPE_diff":raw_MAPE - round(df_experiment_log_cuts.MAPE.drop([0], axis=0).mean(),2),
                              "MAE_leaves":round(df_experiment_log_cuts.MAE.drop([0], axis=0).mean(),2),
                              "MSE_leaves":round(df_experiment_log_cuts.MSE.drop([0], axis=0).mean(),2),
                              "RMSE_leaves":round(df_experiment_log_cuts.RMSE.drop([0], axis=0).mean(),2),
                              "MAPE_leaves":round(df_experiment_log_cuts.MAPE.drop([0], axis=0).mean(),2)
                              })
                  print(f'{file}-{criteria}-{separator_name} salvo no wandb')
                except Exception as e:
                  print('Erro na execução do loop', e)
                finally:
                  run.finish()
