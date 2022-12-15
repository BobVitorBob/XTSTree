from XTSTree.XTSTree import XTSTree
from collections.abc import Iterable
import random


class XTSTreeRandomCut(XTSTree):
  
  def __init__(self, stop_condition: str='depth', stop_val=2, max_iter=1000, min_dist=30):
    super().__init__(stop_condition=stop_condition, stop_val=stop_val, max_iter=max_iter, min_dist=min_dist)

  def _find_cut(self, series: Iterable, params: dict):
    cut = random.randint(self.min_dist, len(series) - self.min_dist)
    return cut, params