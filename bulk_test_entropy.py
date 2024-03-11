import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
import re
import os
from XTSTree.XTSTree import XTSTree
from XTSTree.XTSTreePageHinkley import XTSTreePageHinkley
from XTSTree.XTSTreeRandomCut import XTSTreeRandomCut
from XTSTree.XTSTreePeriodicCut import XTSTreePeriodicCut

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
  mean_by_cut, n_items, tot_depth = model.calc_mean_entropy_gain_by_cut()
  return model, mean_by_cut, n_items, tot_depth

# --------------------------------------------------------------------------------------------

output = []
par_files = next(os.walk('./datasets/umidrelmed2m'))[1]
for par_file in par_files:
  child_files = next(os.walk(f'datasets/umidrelmed2m/{par_file}'))[2]
  expected_len = (int(re.sub("[^0-9]", "", par_file)) * 96)
  for rep in range(3):
    for child_file in child_files:
      series = treat_or_discard_series(
        load_series(file_name=f'{par_file}/{child_file}'),
        perc_cut=0.5,
        min_len=expected_len
      )
      if series is False:
        print(f'Série {child_file} rejeitada')
        continue
      
      print(f'Repetição {rep}, Arquivo {child_file}')
      models = [
        ('PageHinkley', XTSTreePageHinkley(stop_condition='adf', stop_val=0, max_iter=100, min_dist=0)),
        ('PeriodicCut', XTSTreePeriodicCut(stop_condition='adf', stop_val=0, max_iter=100, min_dist=0)),
        ('RandomCut_1', XTSTreeRandomCut(stop_condition='adf', stop_val=0, max_iter=100, min_dist=0)),
      ]

      for name, model in models:
        fitted_model, mean_by_cut, n_items, tot_depth = fit_model(model=model, series=series)
        segments = fitted_model.cut_series(series)
        output.append({
          'nome': f'{name}_{rep}_{child_file}',
          'model': name,
          'file': child_file[:-4],
          'ganho médio de entropia por corte': mean_by_cut,
          'numero de segmentos': len(segments),
        })
        pd.DataFrame(output).to_csv('./resultados_entropy.csv', index=False)
        print(f'Terminei o {name}, Repetição {rep}')