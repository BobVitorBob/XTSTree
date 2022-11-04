from collections.abc import Iterable
from river.drift import PageHinkley
from Structures.Tree import Tree, TreeNode
from Separators.Separator import Separator

class SeparatorPageHinkley(Separator):
	
	def __init__(self, stop_condition: str='depth', stop_val=8, max_iter=1000, min_instances: int=30, delta: float=0.005, starting_threshold: float=50.0, alpha: float=1 - 0.0001, mode: str="both"):
		self.min_instances = min_instances
		self.delta = delta
		self.threshold = starting_threshold
		self.alpha = alpha
		self.mode = mode
		super().__init__(stop_condition=stop_condition, stop_val=stop_val, max_iter=max_iter)
		
	def create_splits(self, series: Iterable):
		if self.stop_condition == 'depth':
			split_tree = Tree()
			threshold = self.threshold
			cuts, threshold = self._find_splits(series, threshold)
			return cuts, threshold
		else:
			raise ValueError(f'Stop condition {self.stop_condition} not supported')
 
	def _find_splits(self, series: Iterable, threshold):
		min_threshold = 0
		max_threshold = -1
		for _ in range(self.max_iter):
			ph = PageHinkley(self.delta, threshold, mode='both')
			cuts = []
			for i, val in enumerate(series):
				ph.update(val)
				if ph.drift_detected:
					cuts.append(i)
					if len(cuts) > self.stop_val:
						break
			if len(cuts) == self.stop_val:
				return cuts, threshold
			elif len(cuts) < self.stop_val:
				print(len(cuts), threshold)
				max_threshold = threshold
				threshold -= (threshold-min_threshold)/2
			elif len(cuts) > self.stop_val:
				print(len(cuts), threshold)
				min_threshold = threshold
				if max_threshold < -1:
					threshold += threshold/2
				else:
					threshold += (max_threshold - threshold)/2
		return cuts, threshold