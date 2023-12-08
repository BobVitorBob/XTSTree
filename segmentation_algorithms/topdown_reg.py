from XTSTree.XTSTree import XTSTree
from collections.abc import Iterable
from segmentation_algorithms.utils import *

class XTSTreeTopDownReg(XTSTree):
  
  def __init__(self, stop_condition: str='depth', stop_val=2, max_iter=1000, min_dist=0):
    super().__init__(stop_condition=stop_condition, stop_val=stop_val, max_iter=max_iter, min_dist=min_dist)

  def _find_cut(self, series: Iterable, params: dict, depth=0):
    m_model, _, _, _ = apply_lr(np.array(range(len(series))), series)

    yhat = m_model.linspace(len(series))[1]
    print(yhat)
    input()
    plot(series, sec_plots=[yhat])
    error = rmse(series, yhat)
    cut = np.argmax(np.positive(series - yhat))
    return cut, params, [np.positive(series - yhat)]