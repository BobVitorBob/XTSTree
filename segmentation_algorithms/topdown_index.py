from XTSTree.XTSTree import XTSTree
from collections.abc import Iterable
from segmentation_algorithms.utils import *
from plot import plot
class XTSTreeTopDownIndex(XTSTree):
  
  def _stop_func(self, series: Iterable, depth:int):
    m_model, _, _, _ = apply_lr(np.arange(len(series)), series)
    yhat = m_model.predict(np.arange(len(series)).reshape(-1, 1))
    error = rmse(series, yhat)
    # plot(series, sec_plots=[yhat], title=f'error: {error}, max: {self.stop_val}, eu {"" if (self.stop_val - error) >= 0 else "não "}paro')
    return self.stop_val - error, series

  def __init__(self, stop_val=2, max_iter=1000, min_dist=0):
    super().__init__(stop_condition=self._stop_func, stop_val=stop_val, max_iter=max_iter, min_dist=min_dist)

  def _find_cut(self, series: Iterable, params: dict, depth=0):
    m_model, _, _, _ = apply_lr(np.arange(len(series)), series)

    yhat = m_model.predict(np.arange(len(series)).reshape(-1, 1))
    cut = np.argmax(np.positive(series - yhat))
    # plot(series, sec_plots=[yhat], divisions=[cut], title=cut)
    return cut, params, np.positive(series - yhat)
