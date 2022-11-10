from collections.abc import Iterable
from typing import Callable, Optional
from statsmodels.tsa.stattools import adfuller
from Structures.Tree import Tree, TreeNode

class XTSTree:
  
  def __init__(self, stop_condition:str='depth', stop_val=3, max_iter=1000, min_dist=50, params:dict={}):
    self.stop_condition = stop_condition
    self.stop_val = stop_val
    self.max_iter = max_iter
    self.min_dist = min_dist
    self.params = params
  
  def _depth_stop_condition(self, series: Iterable, depth:int):
    return depth >= (self.stop_val - 1)
  
  def _adf_stop_condition(self, series: Iterable, depth:int):
    adf = adfuller(series)
    return adf[1] < self.stop_val
  
  # Cria a árvore e acha os splits para uma série
  def create_splits(self, series: Iterable):
    if self.stop_condition == 'depth':
      stop_func=self._depth_stop_condition
    elif self.stop_condition == 'adf':
      stop_func=self._adf_stop_condition
    else:
      raise ValueError(f'Stop condition {self.stop_condition} not supported')
    xtstree = Tree()
    xtstree.root = self._recursive_tree(series, stop_func=stop_func, params=self.params)
    return xtstree

  # Função recursiva para encontrar os nós e criar a árvore
  def _recursive_tree(self, series: Iterable, stop_func: Callable[[Iterable, int], bool], params: dict, curr_depth=0):
    node = TreeNode(series)
    # Se a é pra parar, retorna o nó com o pedaço da série
    if stop_func(series, curr_depth):
      return node
    
    # Achando a posição de corte e pegando os parâmetros da função de corte
    # Isso permite que a função de corte altere os parâmetros pra chamada dos próximos nós para otimizar os cortes
    cut_pos, params = self._find_cut(series=series, params=params)
    
    # Se achou uma posição de corte, corta a série e procura na esquerda e na direita
    if cut_pos >= 0:
      node.left = self._recursive_tree(series[:cut_pos], stop_func=stop_func, params=params, curr_depth=curr_depth+1)
      node.right = self._recursive_tree(series[cut_pos:], stop_func=stop_func, params=params, curr_depth=curr_depth+1)
      
    # Retorna o nó
    return node
    
  # Função que encontra a posição de corte, única para cada método de corte
  def _find_cut(self, series: Iterable, params: dict) -> (int, dict):
    pass
    
