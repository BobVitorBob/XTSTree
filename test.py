from Separators.SeparatorPageHinkley import SeparatorPageHinkley
from plot import plot
import math
import time
from multiprocessing import Pool
import pandas as pd
import numpy as np
import random
from statsmodels.tsa.arima.model import ARIMA as statsARIMA
from statsforecast.arima import AutoARIMA
from collections.abc import Iterable

window_size = 96
# Multiplicador pra não deixar o desvio muito pequeno
# No cusum os limites recomendados são 4 desvios, com a folga de meio desvio dá mais ou menos isso
s = 3.2
min_std = 0
max_std = 15
# Valor inicial caso tenha anomalia desde o começo
std = min_std

series_path = '../../../Dados/por estacao/23025122/export_automaticas_23025122_umidrelmed2m.csv'

def seq_data(data, window_size=96):
  data_X = []
  data_Y = []
  for i in range(len(data) - (len(data) % window_size) - window_size):
    data_X.append(data[i:i+window_size])
    try:
      data_Y.append(data[i+window_size])
    except:
      (i+window_size)
  return np.array(data_X), np.array(data_Y)

from statsmodels.tsa.seasonal import seasonal_decompose

series_len = 15 * 96

# series = pd.read_csv(series_path, nrows=series_len)['umidrelmed2m']
series = pd.read_csv(series_path)['umidrelmed2m']
# series = [(math.sin(x/100)) for x in range(1000)] + [(math.sin(x/10)) for x in range(1000)]


series = np.array([series[i] if np.isfinite(series[i]) else series[i-1] for i in range(len(series))])

# print('Valor p (> 0.05, não estacionária, < 0.05 é estacionária) ', adfuller(series)[1])
# reshaped_data, targets = seq_data(series, window_size=window_size)
# standart_d = [max(min(np.std(x) * s, max_std), min_std) for x in reshaped_data]

series_diff = np.diff(series)

plot(series)
print('Criando splits')

adf = 0.05

t = time.perf_counter()
sep = XTSTreePageHinkley(stop_condition='adf', stop_val=adf, min_dist=30)

xtstree = sep.create_splits(series)
print(f'Splits achados em {time.perf_counter() - t}: ')

leaves = [len(leaf) for leaf in xtstree.get_leaves()] 

cuts = [sum([leaf for leaf in leaves[:i+1]]) for i in range(len(leaves))][:-1]

print(cuts)
plot(series, divisions=cuts, title=f'Cortes, threshold do adf {adf}')