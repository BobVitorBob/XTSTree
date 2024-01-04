import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike

import re

from segmentation_algorithms.utils import *

from time import perf_counter 

import os

from XTSTree.XTSTree import XTSTree
from segmentation_algorithms.topdown_index import XTSTreeTopDownIndex
from segmentation_algorithms.topdown_reg import XTSTreeTopDownReg
from XTSTree.XTSTreePageHinkley import XTSTreePageHinkley
from XTSTree.XTSTreeRandomCut import XTSTreeRandomCut
from XTSTree.XTSTreePeriodicCut import XTSTreePeriodicCut

from pysr import *

def get_regressor(
    criteria='best',
    output_file='test',
    pop_n=10,
    pop_size=30,
    iterations=3,
    max_complexity=40,
    binary_operators=['+', '-', '*', '/', 'pow'],
    unary_operators=['sqrt', 'sin'],
    verbosity=0,
    early_stop_condition=None
  ):
  """
  constraints é um dicionário com os operadores unários e binários e uma tupla com a complexidade máxima dos argumentos.
  ver o que o should_simplify faz, aparentemente simplifica a equação, mas ver como ele faz isso
  usar random_state pra garantir mesmos resultados
  """
  return PySRRegressor(
    binary_operators=binary_operators,
    unary_operators=unary_operators,
    maxsize=max_complexity,
    niterations=iterations,
    populations=pop_n,
    population_size=pop_size,
    progress=False,
    model_selection=criteria,
    equation_file=f'./symbreg_objects/{output_file}.csv',
    verbosity = verbosity,
    temp_equation_file=False,
    early_stop_condition=early_stop_condition
  )

def treat_or_discard_series(series, perc_cut=0.5, min_len=96*5) -> bool:
  '''
  Retorna a série tratada se a porcentagem de NaNs for menor que perc_cut, e retorna Falso caso contrário 
  '''
  if len(series) < min_len:
    return False
  if (sum(np.isnan(series))/len(series)) < perc_cut:
    return np.where(np.isnan(series), 0, series)
  else:
    return False

def load_series(file_name):
  try:
    df = pd.read_csv(f'./datasets/umidrelmed2m/{file_name}')
    return df[df.columns[-1]]
  except Exception as e:
    print(f'Erro ao carregar a série: {file_name}')
    raise e

def fit_model(model: XTSTree, series: ArrayLike):  
  model = model.create_splits(series)
  model.calculate_entropy_gain()
  model.calc_mean_entropy_gain_by_cut()
  
  return model

def calc_error_lag(series, lags=[]):
  error_lag = {}
  # lags de 1, 6, 12 e 24 horas
  for lag in lags:
    X, y = group_data(series, lag)
    model_lr_lag, _, _, _ = apply_lr(X, y)  
    yhat = model_lr_lag.predict(X)
    error_lag[f'{lag}'] = rmse(series[lag:], yhat)
  return error_lag

def calc_error_index(series):
  model_lr_index, _, _, _ = apply_lr(np.arange(len(series)), series)
  yhat = model_lr_index.predict(np.arange(len(series)).reshape(-1, 1))  
  return rmse(series, yhat)

# --------------------------------------------------------------------------------------------
output = []
par_files = next(os.walk('./datasets/umidrelmed2m'))[1]
for par_file in par_files:
  child_files = next(os.walk(f'datasets/umidrelmed2m/{par_file}'))[2]
  expected_len = (int(re.sub("[^0-9]", "", par_file)) * 96)
  for child_file in child_files:
    series = treat_or_discard_series(
      load_series(file_name=f'{par_file}/{child_file}'),
      perc_cut=0.5,
      min_len=expected_len
    )
    if series is False:
      print(f'Série {child_file} rejeitada')
      continue
    error_lag = calc_error_lag(series, [4, 24, 48, 96])
    error_index = calc_error_index(series)
    models = [
      ('PageHinkley', XTSTreePageHinkley(stop_condition='adf', stop_val=0, max_iter=100, min_dist=0)),
      
      ('PeriodicCut', XTSTreePeriodicCut(stop_condition='adf', stop_val=0, max_iter=100, min_dist=0)),
      
      ('RandomCut_1', XTSTreeRandomCut(stop_condition='adf', stop_val=0, max_iter=100, min_dist=0)),
      ('RandomCut_2', XTSTreeRandomCut(stop_condition='adf', stop_val=0, max_iter=100, min_dist=0)),
      ('RandomCut_3', XTSTreeRandomCut(stop_condition='adf', stop_val=0, max_iter=100, min_dist=0)),
      ('RandomCut_4', XTSTreeRandomCut(stop_condition='adf', stop_val=0, max_iter=100, min_dist=0)),
      ('RandomCut_5', XTSTreeRandomCut(stop_condition='adf', stop_val=0, max_iter=100, min_dist=0)),
      
      *[(f'TopDownReg_{lag}_25', XTSTreeTopDownReg(stop_val=error*0.25, max_iter=100, min_dist=0, lag=lag)) for lag, error in error_lag.items()],
      *[(f'TopDownReg_{lag}_50', XTSTreeTopDownReg(stop_val=error*0.50, max_iter=100, min_dist=0, lag=lag)) for lag, error in error_lag.items()],
      *[(f'TopDownReg_{lag}_75', XTSTreeTopDownReg(stop_val=error*0.75, max_iter=100, min_dist=0, lag=lag)) for lag, error in error_lag.items()],
      
      ('TopDownIndex_25', XTSTreeTopDownIndex(stop_val=error_index*0.25, max_iter=100, min_dist=0)),
      ('TopDownIndex_50', XTSTreeTopDownIndex(stop_val=error_index*0.50, max_iter=100, min_dist=0)),
      ('TopDownIndex_75', XTSTreeTopDownIndex(stop_val=error_index*0.75, max_iter=100, min_dist=0)),
    ]

    for rep in range(1):
      print(f'Repetição {rep}')
      model = get_regressor()
      indexes = np.array([[i] for i, _ in enumerate(series)])
      t = perf_counter()
      model.fit(indexes, series)
      end_t = perf_counter() - t

      complexity_full = model.get_best()["complexity"]
      prediction_full = model.predict(indexes)
      output.append({
        'nome': f'Full_{rep}_{child_file}',
        'model': 'full',
        'file': child_file,
        'MAE (erro entre a série inteira e a predição de todos os segmentos)': mae(series, prediction_full),
        'RMSE (erro entre a série inteira e a predição de todos os segmentos)': rmse(series, prediction_full),
        'complexidade (média dos segmentos)': complexity_full,
        'desvio padrão complexidade': 0,
        'tempo': end_t,
        'numero de segmentos': 1,
      })
      pd.DataFrame(output).to_csv('./resultados.csv', index=False)
      print('Terminou o completo')
      for name, model in models[:1]:
        segments = model.cut_series(series)
        try:
          y_hat = []
          complexities = []
          time = perf_counter()
          for segment in segments:
            modelo = get_regressor()
            indexes = np.array([[i] for i, _ in enumerate(segment)])
            modelo.fit(indexes, segment)
            prediction = modelo.predict(indexes)
            y_hat = y_hat + list(prediction)
            complexities.append(modelo.get_best()["complexity"])
          end_t = perf_counter() - time
          output.append({
            'nome': f'{name}_{rep}_{child_file}',
            'model': name,
            'file': child_file,
            'MAE (erro entre a série inteira e a predição de todos os segmentos)': mae(series, y_hat),
            'RMSE (erro entre a série inteira e a predição de todos os segmentos)': rmse(series, y_hat),
            'complexidade (média dos segmentos)': np.mean(complexities),
            'desvio padrão complexidade': np.std(complexities),
            'tempo': end_t,
            'numero de segmentos': len(segments),
          })
          pd.DataFrame(output).to_csv('./resultados.csv', index=False)
          print(f'Terminei o {name}, Repetição {rep}')
        except Exception as e:
          print(f'Erro no PySR durante a execução nos segmentos')
          print(f'{name}, Repetição {rep}')
          print(f'Tamanho do segmento: {len(segment)}')
          print(f'Equação: {modelo.get_best()["equation"]}')
          print(f'Erro: {e}')
