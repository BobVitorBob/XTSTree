import numpy as np
from plot import plot
import math
from pysr import PySRRegressor
from XTSTree.XTSTreePageHinkley import XTSTreePageHinkley
import pandas as pd
import time

import random

files_path = 'toy_files'

def get_regressor(criteria, output_file, pop_n, pop_size, iterations, path):
    return PySRRegressor(
        binary_operators=['+', '-', '*', '/', 'pow'],
        unary_operators=['neg', 'exp', 'abs', 'log', 'sqrt', 'sin', 'tan', 'sinh', 'sign'],
        niterations=iterations,
        populations=pop_n,
        population_size=pop_size,
        progress=False,
        model_selection=criteria,
        equation_file=f'{path}/symbreg_objects/{output_file}.csv',
        verbosity=0,
        temp_equation_file=False
    )
    
def evaluate_ts(current_ts, model):
    X = []
    y = []
    for i, val in enumerate(current_ts):
        X.append([i])
        y.append(val)

    model.fit(X, y, variable_names=['index'])

    yhat = model.predict(X)
    perf_mae = round(mae(y, yhat),2)
    perf_mse = round(mse(y, yhat),2)
    perf_rmse = round(rmse(y, yhat),2)
    perf_mape = round(mape(y, yhat),2)

    return model, model.get_best()['lambda_format'](np.array(X)), perf_mae, perf_mse, perf_rmse, perf_mape, model.get_best()['complexity'], model.get_best()['equation']
  
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

list_XTSTree = [
  ['PageHinkley_adf', XTSTreePageHinkley(stop_condition='adf', stop_val=0, min_dist=40)],
  # ['RandomCut_adf', XTSTreeRandomCut(stop_condition='adf', stop_val=0, min_dist=0)],
  # ['PeriodicCut_adf', XTSTreePeriodicCut(stop_condition='adf', stop_val=0, min_dist=0)],
]

series = np.concatenate([
	np.array([random.uniform(-0.2, 0.2) for _ in range(200)]) + np.sin([i/10 for i in range(200)]),
	np.array([random.uniform(-0.2, 0.2) for _ in range(200)]) + np.arange(0, 5, 5/200)+np.sin([i/10 for i in range(200)]),
	np.array([random.uniform(-0.2, 0.2) for _ in range(200)]) + 2*np.sin([i/10 for i in range(200)]),
	np.array([random.uniform(-0.2, 0.2) for _ in range(200)]) + np.sin([i/10 for i in range(200)]),
])
plot(series, show=False, save=True, img_name=f'{files_path}/images/series.jpeg', show_axis=(True, False))
for criteria in ['accuracy', 'best']:


  for tree_name, xtstree in list_XTSTree:
    print('Aplicando SR série toda,', criteria)
    t = time.perf_counter()
    model, yhat, series_MAE, series_MSE, series_RMSE, series_MAPE, series_complexity, formula = evaluate_ts(
                series,
                get_regressor(criteria, 'toy_full', 20, 40, 5, files_path)
              )
    t_diff = time.perf_counter() - t

    plot(series, show=False, save=True, img_name=f'{files_path}/images/series_sr_{criteria}.jpeg', sec_plots=[yhat], show_axis=(True, False))
    print(formula)
    print(f'series_MAE: {series_MAE}', f'series_MSE: {series_MSE}', f'series_RMSE: {series_RMSE}', f'series_MAPE: {series_MAPE}', f'series_complexity: {series_complexity}')
    experiment_log_cuts = [[
      0,
      len(series),
      series_MAE,
      series_MSE,
      series_RMSE,
      series_MAPE,
      formula,
      series_complexity,
      criteria,
      t_diff
    ]]

    print('---------------------------------------XTSTree---------------------------------------')
    t = time.perf_counter()
    xtstree = xtstree.create_splits(series)
    t_split = time.perf_counter() - t
    heatmap = xtstree.get_heatmap()

    cuts = xtstree.cut_points()
    
    plot(series, divisions=cuts, show=False, save=True, img_name=f'{files_path}/images/series_cuts_{criteria}.jpeg', title=f'Segments with {tree_name}', show_axis=(True, False))
    plot(series, divisions=cuts, show=False, save=True, img_name=f'{files_path}/images/series_cuts_{criteria}_heatmap.jpeg', title=f'Heatmap for {tree_name}', color_gradient=heatmap, color_pallete='viridis', show_axis=(True, False))

    plot_cuts = []
    for start, finish in zip([0, *cuts], [*cuts, len(series)]):

      t_cut = time.perf_counter()
      model, yhat, leaf_MAE, leaf_MSE, leaf_RMSE, leaf_MAPE, leaf_complexity, formula = evaluate_ts(
        series[start:finish],
        get_regressor(
          criteria,
          f'toy_example_{start}-{finish}',
          20,
          40,
          5,
          files_path
        )
      )
      t_cut_diff = time.perf_counter() - t_cut
      plot_cuts.append(yhat)
      experiment_log_cuts.append([
        start,
        finish,
        leaf_MAE,
        leaf_MSE,
        leaf_RMSE,
        leaf_MAPE,
        formula,
        leaf_complexity,
        criteria,
        t_cut_diff
      ])

    plot(series, divisions=cuts, show=False, save=True, img_name=f'{files_path}/images/series_sr_on_cuts_{criteria}.jpeg', title=f'Segments with {tree_name}', sec_plots=[np.concatenate(plot_cuts).ravel().tolist()], show_axis=(True, False))
    plot(series, divisions=cuts, show=False, save=True, img_name=f'{files_path}/images/series_sr_on_cuts_{criteria}.jpeg', title=f'Heatmap for {tree_name}', color_gradient=heatmap, color_pallete='viridis', sec_plots=[np.concatenate(plot_cuts).ravel().tolist()], show_axis=(True, False))

    df_experiment_log_cuts = pd.DataFrame(experiment_log_cuts)

    df_experiment_log_cuts.columns = [
      "Start",
      "Finish",
      "MAE",
      "MSE",
      "RMSE",
      "MAPE",
      "Equation",
      "Complexity",
      "Criteria",
      "Time"
    ]
    df_experiment_log_cuts.to_csv(f'{files_path}/results_0_{criteria}.csv', index=False)
    for metric in ['MAE', 'MSE', 'RMSE', 'MAPE', 'Complexity', 'Time']:
      print(f'Mean {metric}', df_experiment_log_cuts[metric].mean())
      print(f'Std {metric}', df_experiment_log_cuts[metric].std())
    
    print(f'Equações: ')
    for eq in experiment_log_cuts:
      print(eq[6])
