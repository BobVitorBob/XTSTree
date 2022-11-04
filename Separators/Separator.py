from collections.abc import Iterable

class Separator:
	
	def __init__(self, stop_condition:str='depth', stop_val=8, max_iter=1000):
		self.stop_condition = stop_condition
		self.stop_val = stop_val
		self.max_iter = max_iter
    
  
	def create_splits(self, series: Iterable):
		pass