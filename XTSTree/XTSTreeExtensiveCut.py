from XTSTree.XTSTree import XTSTree
from collections.abc import Iterable

class XTSTreeExtensiveCut(XTSTree):
  
  def __init__(self, stop_condition: str='depth', stop_val=2, max_iter=1000, min_dist=0):
    super().__init__(stop_condition=stop_condition, stop_val=stop_val, max_iter=max_iter, min_dist=min_dist)

  def _find_cut(self, series: Iterable, params: dict, depth=0):
    max_points = -1
    max_i = -1
    for i in range(1, len(series)):
      i_points = self.stop_func(series[:i], depth) + self.stop_func(series[i:], depth)
      if i_points > max_points:
        max_points = i_points
        max_i = i
    return max_i, params