from Structures.Tree import Tree, TreeNode
from collections.abc import Iterable
from typing import Tuple, List
from statsmodels.tsa.stattools import adfuller

class XTSTree:
  
  def __init__(self, stop_condition:str='depth', stop_val=3, max_iter=1000, min_dist=50, params:dict={}):
    if stop_condition == 'depth':
      self.stop_func=self._depth_stop_condition
    elif stop_condition == 'adf':
      self.stop_func=self._adf_stop_condition
    else:
      raise ValueError(f'Stop condition {stop_condition} not supported')
    self.stop_val = stop_val
    self.max_iter = max_iter
    self.min_dist = min_dist
    self.params = params
    self.tree = Tree()
  
  def _depth_stop_condition(self, series: Iterable, depth:int):
    return (self.stop_val - 1) - depth
  
  def _adf_stop_condition(self, series: Iterable, depth=0):
    adf_test = adfuller(series)
    return adf_test[1] - self.stop_val
  
  # Cria a árvore e acha os splits para uma série
  def create_splits(self, series: Iterable):
    self.tree.root = self._recursive_tree(series, params=self.params)
    return self.tree

  # Função recursiva para encontrar os nós e criar a árvore
  def _recursive_tree(self, series: Iterable, params: dict, curr_depth=0):
    if self.stop_func(series, curr_depth) > 0 or len(series) < (self.min_dist * 2):
      return None
    # Achando a posição de corte e pegando os parâmetros da função de corte
    # Isso permite que a função de corte altere os parâmetros pra chamada dos próximos nós para otimizar os cortes
    cut_pos, params = self._find_cut(series=series, params=params)

    node = TreeNode(cut_pos)
    # Se achou uma posição de corte, corta a série e procura na esquerda e na direita
    if cut_pos >= 0:
      node.left = self._recursive_tree(series[:cut_pos], params=params, curr_depth=curr_depth+1)
      node.right = self._recursive_tree(series[cut_pos:], params=params, curr_depth=curr_depth+1)
      
    # Retorna o nó
    return node
    
  # Função que encontra a posição de corte, única para cada método de corte
  def _find_cut(self, series: Iterable, params: dict) -> Tuple[int, dict]:
    pass

  def apply_on_leaves(self, function, series: Iterable) -> List:
    return list(map(function, self.tree.cut_series(series)))

  def to_list(self) -> List:
    return self.tree.to_list()

  def cut_series(self, series: Iterable):
    return XTSTree._get_cuts(self.tree.root, series)
      
  def cut_points(self):
    return XTSTree._get_cut_points(self.tree.root)

  def depth(self):
    return self.tree.depth()
  
  def cut_series_by_depth(self, series):
    cuts_by_depth = self.get_cuts_by_depth()
    cut_series = {}
    for depth in cuts_by_depth.keys():
      cuts = sorted([cut for i in range(depth+1) for cut in cuts_by_depth[i]])
      cut_series[depth] = [series[start:end] for start, end in zip([0] + cuts, cuts + [len(series)])]
    return cut_series

  def get_cuts_by_depth(self):
    return XTSTree._get_cuts_by_depth(node=self.tree.root, depth=0)
  
  @staticmethod
  def _get_cuts_by_depth(node: TreeNode, depth: int):
    if node is None:
      return {}
    cuts_dict = {depth: [node.cont]}
    for key, val in XTSTree._get_cuts_by_depth(node=node.left, depth=depth + 1).items():
      if key in cuts_dict:
        cuts_dict[key] += val
      else:
        cuts_dict[key] = val
    for key, val in XTSTree._get_cuts_by_depth(node=node.right, depth=depth + 1).items():
      if key in cuts_dict:
        cuts_dict[key] += [cut + node.cont for cut in val]
      else:
        cuts_dict[key] = [cut + node.cont for cut in val]
    
    return cuts_dict

  @staticmethod
  def _get_cuts(node: TreeNode, series: Iterable):
    if node is None:
      return [series]
    return [*XTSTree._get_cuts(node.left, series[:node.cont]), *XTSTree._get_cuts(node.right, series[node.cont:])]

  @staticmethod
  def _get_cut_points(node: TreeNode) -> List:
    if node is None:
      return []
    return [*XTSTree._get_cut_points(node.left), node.cont, *[cut + node.cont for cut in XTSTree._get_cut_points(node.right)]]