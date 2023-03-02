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

    return model, model.get_best()['lambda_format'](np.array(X)), perf_mae, perf_mse, perf_rmse, perf_mape, model.get_best()['complexity']

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


def get_regressor(criteria, output_file, iterations, path):
    return PySRRegressor(
        binary_operators=['+', '-', '*', '/', 'pow'],
        unary_operators=['neg', 'exp', 'abs', 'log', 'sqrt', 'sin', 'tan', 'sinh', 'sign'],
        niterations=iterations,
        populations=30,
        population_size=60,
        progress=False,
        model_selection=criteria,
        equation_file=f'{path}symbreg_objects/{output_file}',
        verbosity = 0,
        temp_equation_file=False
        )

param_dataset = sys.argv[1] #numero de dias
wandb_project_name = sys.argv[2]
wandb_entity = sys.argv[3]


#remover depois
param_path = "./test/"
param_niterations = 20

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
                ['PageHinkley_adf', XTSTreePageHinkley(stop_condition='adf', stop_val=0.05, min_dist=0)],
                ['PageHinkley_depth', XTSTreePageHinkley(stop_condition='depth', stop_val=3, min_dist=0)],
                ['RandomCut_adf', XTSTreeRandomCut(stop_condition='adf', stop_val=0.05, min_dist=0)],
                ['RandomCut_depth', XTSTreeRandomCut(stop_condition='depth', stop_val=3, min_dist=0)],
                ['PeriodicCut_adf', XTSTreePeriodicCut(stop_condition='adf', stop_val=0.05, min_dist=0)],
                ['PeriodicCut_depth', XTSTreePeriodicCut(stop_condition='depth', stop_val=3, min_dist=0)],
                ]

df_experiment_log = pd.DataFrame(
  columns=[
    # Exp info
    "File",
    "XTSTree",
    "Criteria",
    "NumIterations",

    # Series Metrics
    "Time Raw Series",
    "MAE Series",
    "MSE Series",
    "RMSE Series",
    "MAPE Series",
    "Complexity Series",

    # Comparative Metrics
    "MAE_diff",
    "MSE_diff",
    "RMSE_diff",
    "MAPE_diff",
    "Complexity_diff",
    "MAE_diff_by_leaf",
    "MSE_diff_by_leaf",
    "RMSE_diff_by_leaf",
    "MAPE_diff_by_leaf",
    "Complexity_diff_by_leaf",

    # XTSTree Metrics
    "Cuts",
    "XTSTree Cut Time",

    # Time
    "Mean_Leaf_Time",
    "Std_Leaf_Time",
    "Min_Leaf_Time",
    "Max_Leaf_Time",
    "Sum_Leaf_Time",

    # SR Error
    "Mean_Leaf_MAE",
    "Mean_Leaf_MSE",
    "Mean_Leaf_RMSE",
    "Mean_Leaf_MAPE",

    "Std_Leaf_MAE",
    "Std_Leaf_MSE",
    "Std_Leaf_RMSE",
    "Std_Leaf_MAPE",

    "Min_Leaf_MAE",
    "Min_Leaf_MSE",
    "Min_Leaf_RMSE",
    "Min_Leaf_MAPE",

    "Max_Leaf_MAE",
    "Max_Leaf_MSE",
    "Max_Leaf_RMSE",
    "Max_Leaf_MAPE",

    "Sum_Leaf_MAE",
    "Sum_Leaf_MSE",
    "Sum_Leaf_RMSE",
    "Sum_Leaf_MAPE",

    # Complexity
    "Mean_Leaf_Complexity",
    "Std_Leaf_Complexity",
    "Min_Leaf_Complexity",
    "Max_Leaf_Complexity",
    "Sum_Leaf_Complexity",
  ]
)

df_experiment_log.to_csv(param_path+f"experiment_log_{param_dataset}.csv", index=False)

for rep in range(1, 5):
    for file in list_files:
        for sep in list_XTSTree:
            separator_name = sep[0]
            separator = sep[1]

            print('Arquivo', file)
            series = pd.read_csv(dir_path+file).dropna()
            plot(series.umidrelmed2m, save=True, show=False, img_name=param_path+"images/"+file+".pdf")

            t = time.perf_counter()
            xtstree = separator.create_splits(series.umidrelmed2m.values)
            t_split = time.perf_counter() - t
            cuts = xtstree.cut_points()
            plot(series.umidrelmed2m, divisions=cuts, title=f'Segments with {separator_name}', save=True, show=False,
                 img_name=param_path+"images/"+file+"_rep"+str(rep)+"_method"+separator_name+"_splits.pdf")

            if len(cuts) == 0:
                print('0 cortes')
                continue
            print("Cuts:", cuts)
            for criteria in list_criteria:

                ### WANDB
                run = wandb.init(project=wandb_project_name, entity=wandb_entity, reinit=True, name=file+"_"+separator_name+"_rep"+str(rep)+"_"+criteria)
                try:
                  print('Avaliando time series inteira')
                  t_raw = time.perf_counter()
                  model, yhat, series_MAE, series_MSE, series_RMSE, series_MAPE, series_complexity = evaluate_ts(series, get_regressor(criteria, f'{criteria}_{separator_name}_{0}_{file}', param_niterations, param_path))
                  t_series = time.perf_counter() - t_raw
                  print('Terminei a série inteira')
                  plot(series.umidrelmed2m, save=True, show=False,
                        img_name=param_path + "images/" + file + "_splits_"+criteria+"_rep"+str(rep)+"_method"+separator_name+"_reg.pdf", sec_plots=[yhat])

                  experiment_log_cuts = [[
                    0,
                    series_MAE,
                    series_MSE,
                    series_RMSE,
                    series_MAPE,
                    model.get_best()['equation'],
                    series_complexity,
                    criteria,
                    param_niterations,
                    t_series
                  ]]
                  plot_cuts = list()
                  print('Avaliando folhas')
                  for start, finish in zip([0, *cuts], [*cuts, len(series.umidrelmed2m.values)]):
                      #print(idx,len(cuts))
                      t_cut = time.perf_counter()
                      model, yhat, leaf_MAE, leaf_MSE, leaf_RMSE, leaf_MAPE, leaf_complexity = evaluate_ts(
                        series.iloc[start:finish, :].copy(), 
                        get_regressor(
                          criteria,
                          f'{criteria}_{separator_name}_{finish}_{file}',
                          param_niterations,
                          param_path
                        )
                      )
                      t_cut_diff = time.perf_counter() - t_cut
                      plot_cuts.append(yhat)
                      experiment_log_cuts.append([
                        finish,
                        leaf_MAE,
                        leaf_MSE,
                        leaf_RMSE,
                        leaf_MAPE,
                        model.get_best()['equation'],
                        leaf_complexity,
                        criteria,
                        param_niterations,
                        t_cut_diff
                      ])

                  plot(series.umidrelmed2m, divisions=cuts, title=f'Segments with {separator_name}', save=True, show=False,
                            img_name=param_path + "images/" + file + "_splits_"+criteria+"_rep"+str(rep)+"_method"+separator_name+"_cuts_reg.pdf", sec_plots=[np.concatenate(plot_cuts).ravel().tolist()])

                  df_experiment_log_cuts = pd.DataFrame(experiment_log_cuts)

                  df_experiment_log_cuts.columns = [
                    "Start",
                    "MAE",
                    "MSE",
                    "RMSE",
                    "MAPE",
                    "Equation",
                    "Complexity",
                    "Criteria",
                    "NumIterations",
                    "Time"
                  ]
                  df_experiment_log_cuts.to_csv(param_path+"logs/"+criteria+"_"+file+"_rep"+str(rep)+"_cuts_log.csv")
                  df_experiment_log_cuts = df_experiment_log_cuts.drop([0], axis=0)
                  # Atualiza o dataframe de log do experimento conforme executa para ter o arquivo caso o experimento quebre
                  df_experiment_log = pd.DataFrame({
                    'File': [file],
                    'XTSTree': [separator_name],
                    'Criteria': [criteria],
                    'NumIterations': [param_niterations],
                    
                    'Time Raw Series': [t_series],
                    'MAE Series': [series_MAE],
                    'MSE Series': [series_MSE],
                    'RMSE Series': [series_RMSE],
                    'MAPE Series': [series_MAPE],
                    'Complexity Series': [series_complexity],
                    
                    'MAE_diff': [series_MAE - round(df_experiment_log_cuts.MAE.mean(),2)],
                    'MSE_diff': [series_MSE - round(df_experiment_log_cuts.MSE.mean(),2)],
                    'RMSE_diff': [series_RMSE - round(df_experiment_log_cuts.RMSE.mean(),2)],
                    'MAPE_diff': [series_MAPE - round(df_experiment_log_cuts.MAPE.mean(),2)],
                    'Complexity_diff': [series_complexity - df_experiment_log_cuts.Complexity],
                    'MAE_diff_by_leaf': [(series_MAE - round(df_experiment_log_cuts.MAE.mean(),2)) / (len(cuts) + 1)],
                    'MSE_diff_by_leaf': [(series_MSE - round(df_experiment_log_cuts.MSE.mean(),2)) / (len(cuts) + 1)],
                    'RMSE_diff_by_leaf': [(series_RMSE - round(df_experiment_log_cuts.RMSE.mean(),2)) / (len(cuts) + 1)],
                    'MAPE_diff_by_leaf': [(series_MAPE - round(df_experiment_log_cuts.MAPE.mean(),2)) / (len(cuts) + 1)],
                    'Complexity_diff_by_leaf': [(series_complexity - df_experiment_log_cuts.Complexity) / (len(cuts) + 1)],
                    
                    'Cuts': [len(cuts)],
                    'XTSTree Cut Time': [t_split],

                    'Mean_Leaf_Time': [round(df_experiment_log_cuts.Time.mean(),2)],
                    'Std_Leaf_Time': [round(df_experiment_log_cuts.Time.std(),2)],
                    'Min_Leaf_Time': [df_experiment_log_cuts.Time.min()],
                    'Max_Leaf_Time': [df_experiment_log_cuts.Time.max()],
                    'Sum_Leaf_Time': [df_experiment_log_cuts.Time.sum()],
                
                    'Mean_Leaf_MAE': [round(df_experiment_log_cuts.MAE.mean(),2)],
                    'Mean_Leaf_MSE': [round(df_experiment_log_cuts.MSE.mean(),2)],
                    'Mean_Leaf_RMSE': [round(df_experiment_log_cuts.RMSE.mean(),2)],
                    'Mean_Leaf_MAPE': [round(df_experiment_log_cuts.MAPE.mean(),2)],

                    'Std_Leaf_MAE': [round(df_experiment_log_cuts.MAE.std(),2)],
                    'Std_Leaf_MSE': [round(df_experiment_log_cuts.MSE.std(),2)],
                    'Std_Leaf_RMSE': [round(df_experiment_log_cuts.RMSE.std(),2)],
                    'Std_Leaf_MAPE': [round(df_experiment_log_cuts.MAPE.std(),2)],

                    'Min_Leaf_MAE': [df_experiment_log_cuts.MAE.min()],
                    'Min_Leaf_MSE': [df_experiment_log_cuts.MSE.min()],
                    'Min_Leaf_RMSE': [df_experiment_log_cuts.RMSE.min()],
                    'Min_Leaf_MAPE': [df_experiment_log_cuts.MAPE.min()],

                    'Max_Leaf_MAE': [df_experiment_log_cuts.MAE.max()],
                    'Max_Leaf_MSE': [df_experiment_log_cuts.MSE.max()],
                    'Max_Leaf_RMSE': [df_experiment_log_cuts.RMSE.max()],
                    'Max_Leaf_MAPE': [df_experiment_log_cuts.MAPE.max()],

                    'Sum_Leaf_MAE': [df_experiment_log_cuts.MAE.sum()],
                    'Sum_Leaf_MSE': [df_experiment_log_cuts.MSE.sum()],
                    'Sum_Leaf_RMSE': [df_experiment_log_cuts.RMSE.sum()],
                    'Sum_Leaf_MAPE': [df_experiment_log_cuts.MAPE.sum()],

                    'Mean_Leaf_Complexity': [round(df_experiment_log_cuts.Complexity.mean(),2)],
                    'Std_Leaf_Complexity': [round(df_experiment_log_cuts.Complexity.std(),2)],
                    'Min_Leaf_Complexity': [df_experiment_log_cuts.Complexity.min()],
                    'Max_Leaf_Complexity': [df_experiment_log_cuts.Complexity.max()],
                    'Sum_Leaf_Complexity': [df_experiment_log_cuts.Complexity.sum()],
                  })
                  df_experiment_log.to_csv(param_path+f"{run_name}.csv", header=None, mode='a', index=False)
                  print(f'{run_name} salvo localmente, subindo para o wandb')
                  ### WANDB
                  wandb.log({
                    # Exp info
                    "File": file,
                    "XTSTree": separator_name,
                    "Criteria": criteria,
                    "NumIterations": param_niterations,

                    # Series Metrics
                    "Time Raw Series": t_series,
                    "MAE Series": series_MAE,
                    "MSE Series": series_MSE,
                    "RMSE Series": series_RMSE,
                    "MAPE Series": series_MAPE,
                    "Complexity Series": series_complexity,

                    # Comparative Metrics
                    "MAE_diff": series_MAE - round(df_experiment_log_cuts.MAE.mean(),2),
                    "MSE_diff": series_MSE - round(df_experiment_log_cuts.MSE.mean(),2),
                    "RMSE_diff": series_RMSE - round(df_experiment_log_cuts.RMSE.mean(),2),
                    "MAPE_diff": series_MAPE - round(df_experiment_log_cuts.MAPE.mean(),2),
                    "Complexity_diff": series_complexity - df_experiment_log_cuts.Complexity.mean(),
                    "MAE_diff_by_leaf": (series_MAE - round(df_experiment_log_cuts.MAE.mean(),2)) / (len(cuts) + 1),
                    "MSE_diff_by_leaf": (series_MSE - round(df_experiment_log_cuts.MSE.mean(),2)) / (len(cuts) + 1),
                    "RMSE_diff_by_leaf": (series_RMSE - round(df_experiment_log_cuts.RMSE.mean(),2)) / (len(cuts) + 1),
                    "MAPE_diff_by_leaf": (series_MAPE - round(df_experiment_log_cuts.MAPE.mean(),2)) / (len(cuts) + 1),
                    "Complexity_diff_by_leaf": (series_complexity - df_experiment_log_cuts.Complexity.mean()) / (len(cuts) + 1),

                    # XTSTree Metrics
                    "Cuts": len(cuts),
                    "XTSTree Cut Time": t_split,

                    # Time
                    "Mean_Leaf_Time": round(df_experiment_log_cuts.Time.mean(),2),
                    "Std_Leaf_Time": round(df_experiment_log_cuts.Time.std(),2),
                    "Min_Leaf_Time": df_experiment_log_cuts.Time.min(),
                    "Max_Leaf_Time": df_experiment_log_cuts.Time.max(),
                    "Sum_Leaf_Time": df_experiment_log_cuts.Time.sum(),

                    # SR Error
                    "Mean_Leaf_MAE": round(df_experiment_log_cuts.MAE.mean(),2),
                    "Mean_Leaf_MSE": round(df_experiment_log_cuts.MSE.mean(),2),
                    "Mean_Leaf_RMSE": round(df_experiment_log_cuts.RMSE.mean(),2),
                    "Mean_Leaf_MAPE": round(df_experiment_log_cuts.MAPE.mean(),2),

                    "Std_Leaf_MAE": round(df_experiment_log_cuts.MAE.std(),2),
                    "Std_Leaf_MSE": round(df_experiment_log_cuts.MSE.std(),2),
                    "Std_Leaf_RMSE": round(df_experiment_log_cuts.RMSE.std(),2),
                    "Std_Leaf_MAPE": round(df_experiment_log_cuts.MAPE.std(),2),

                    "Min_Leaf_MAE": df_experiment_log_cuts.MAE.min(),
                    "Min_Leaf_MSE": df_experiment_log_cuts.MSE.min(),
                    "Min_Leaf_RMSE": df_experiment_log_cuts.RMSE.min(),
                    "Min_Leaf_MAPE": df_experiment_log_cuts.MAPE.min(),

                    "Max_Leaf_MAE": df_experiment_log_cuts.MAE.max(),
                    "Max_Leaf_MSE": df_experiment_log_cuts.MSE.max(),
                    "Max_Leaf_RMSE": df_experiment_log_cuts.RMSE.max(),
                    "Max_Leaf_MAPE": df_experiment_log_cuts.MAPE.max(),

                    "Sum_Leaf_MAE": df_experiment_log_cuts.MAE.sum(),
                    "Sum_Leaf_MSE": df_experiment_log_cuts.MSE.sum(),
                    "Sum_Leaf_RMSE": df_experiment_log_cuts.RMSE.sum(),
                    "Sum_Leaf_MAPE": df_experiment_log_cuts.MAPE.sum(),

                    # Complexity
                    "Mean_Leaf_Complexity": round(df_experiment_log_cuts.Complexity.mean(),2),
                    "Std_Leaf_Complexity": round(df_experiment_log_cuts.Complexity.std(),2),
                    "Min_Leaf_Complexity": df_experiment_log_cuts.Complexity.min(),
                    "Max_Leaf_Complexity": df_experiment_log_cuts.Complexity.max(),
                    "Sum_Leaf_Complexity": df_experiment_log_cuts.Complexity.sum(),
                  })
                  print(f'{file}-{criteria}-{separator_name} salvo no wandb')
                except Exception as e:
                  print(f'Erro no loop, {separator_name}, {criteria}, {file}, {e}')
                finally:
                  run.finish()
