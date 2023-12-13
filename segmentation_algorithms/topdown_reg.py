from XTSTree.XTSTree import XTSTree
from collections.abc import Iterable
from typing import Tuple
from segmentation_algorithms.utils import *
from plot import plot

def group_data(data: Iterable, window_size=96) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separa o vetor data em pares de vetores X de entrada e valor de predição y
    data: Conjunto de dados
    window_size: Tamanho dos vetores de dados em X
    Retorna o vetor X e Y
    """
    data_X = []
    data_Y = []
    len_data = len(data)
    for i in range(len_data - window_size):
        data_X.append(data[i:i + window_size])
        data_Y.append(data[i + window_size])
    return np.array(data_X), np.array(data_Y)

class XTSTreeTopDownReg(XTSTree):
  
  def _stop_func(self, series: Iterable, depth:int):
    if len(series) <= self._lag:
      return float('inf'), series
    X, y = group_data(series, self._lag)
    m_model, _, _, _ = apply_lr(X, y)
    yhat = m_model.predict(X)
    error = rmse(series[self._lag:], yhat)
    # plot(series[self._lag:], sec_plots=[yhat], title=f'error: {error}, max: {self.stop_val}, eu {"" if (self.stop_val - error) >= 0 else "não "}paro')
    return self.stop_val - error, series

  def __init__(self, stop_val=2, max_iter=1000, min_dist=0, lag=4):
    super().__init__(stop_condition=self._stop_func, stop_val=stop_val, max_iter=max_iter, min_dist=min_dist)
    self._lag = lag

  def _find_cut(self, series: Iterable, params: dict, depth=0):
    X, y = group_data(series, self._lag)
    # print(X, y, series)
    m_model, _, _, _ = apply_lr(X, y)

    yhat = m_model.predict(X)
    cut = np.argmax(np.positive(series[self._lag:] - yhat))+self._lag
    # plot(series[self._lag:], sec_plots=[yhat], divisions=[cut], title=cut)
    return cut, params, np.positive(series[self._lag:] - yhat)
