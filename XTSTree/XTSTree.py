from Structures.Tree import Tree, TreeNode
from collections.abc import Iterable, Callable
from typing import Tuple, List
from statsmodels.tsa.stattools import adfuller, kpss
import numpy as np
from sklearn.linear_model import LinearRegression

def rmse(y, y_hat):
  return np.sqrt(np.mean(np.square(y - y_hat)))

def apply_lr(X, y, silent=True):
  X = np.array(X)
  if X.ndim == 1:
    X = X.reshape(-1, 1)
  reg = LinearRegression().fit(X, y)
  if not silent:
    print('score', reg.score(X, y))
    print('coef_', reg.coef_)
    print('intercept_', reg.intercept_)
  return reg, reg.score(X, y), reg.coef_, reg.intercept_

class XTSTree:
  
  def __init__(self, stop_condition:(str | Callable[[Iterable, float], (float, Iterable)])='depth', stop_val=3, max_iter=1000, min_dist=0, params:dict={}):
    if callable(stop_condition):
      self.stop_func=stop_condition
    elif stop_condition == 'depth':
      self.stop_func=self._depth_stop_condition
    elif stop_condition == 'adf':
      self.stop_func=self._adf_stop_condition
    elif stop_condition == 'kpss':
      self.stop_func=self._kpss_stop_condition
    elif stop_condition == 'adf_kpss':
      self.stop_func=self._adf_kpss_stop_condition
    elif stop_condition == 'lr_error':
      self.stop_func=self._lr_error_stop_condition
    else:
      raise ValueError(f'Stop condition {stop_condition} not supported')
    self.stop_val = stop_val
    self.max_iter = max_iter
    self.min_dist = max(min_dist, 0)
    self.params = params
    self.tree = Tree()
  
  def _depth_stop_condition(self, series: Iterable, depth:int):
    return depth - (self.stop_val - 1), series
  
  def _lr_error_stop_condition(self, series: Iterable, depth:int):
    m_model, _, _, _ = apply_lr(np.array(range(len(series))), series)
    y_hat = m_model.predict(np.array(range(len(series))).reshape(-1, 1))
    error = rmse(series, y_hat)
    return self.stop_val - error, series
  
  def _adf_kpss_stop_condition(self, series: Iterable, depth:int):
    try:
      adf_test = adfuller(series)
      print('adf')
      print(adf_test)
    except ValueError as e:
      # Série pequena demais
      print('Série pequena demais para adf, deve terminar o corte')
      # Retorna 0 pra parar os cortes e recompensar o mínimo possível a folha
      return 0, series
    try:
      kpss_test = kpss(series)
      print('kpss')
      print(kpss_test)
    except ValueError as e:
      # Série pequena demais
      print('Série pequena demais para kpss, deve terminar o corte')
      # Retorna 0 pra parar os cortes e recompensar o mínimo possível a folha
      return 0, series
    kpss_test_result = kpss_test[3]['5%'] - kpss_test[0] - self.stop_val
    adf_test_result = adf_test[4]['5%'] - adf_test[0] - self.stop_val
    
    if kpss_test_result * adf_test_result < 0:
      new_series = np.diff(np.array(series))
      adf_test = adfuller(series)
      kpss_test = kpss(series)
      kpss_test_result = kpss_test[3]['5%'] - kpss_test[0] - self.stop_val
      adf_test_result = adf_test[4]['5%'] - adf_test[0] - self.stop_val
    else:
      new_series = series
    
    return min(kpss_test_result, adf_test_result), new_series
  
  def _kpss_stop_condition(self, series: Iterable, depth=0):
    try:
      kpss_test = kpss(series)
    except ValueError as e:
      # Série pequena demais
      print('Série pequena demais para kpss, deve terminar o corte')
      # Retorna 0 pra parar os cortes e recompensar o mínimo possível a folha
      return 0, series
    return kpss_test[0] - kpss_test[3]['5%'] - self.stop_val, series
  
  def _adf_stop_condition(self, series: Iterable, depth=0):
    try:
      adf_test = adfuller(series, regression='c')
    except ValueError as e:
      # Série pequena demais
      print('Série pequena demais para adf, deve terminar o corte')
      # Retorna 0 pra parar os cortes e recompensar o mínimo possível a folha
      return 0, series
      
    return adf_test[0] - adf_test[4]['5%'] + self.stop_val, series
  
  # Cria a árvore e acha os splits para uma série
  def create_splits(self, series: Iterable):
    self.tree.root = self._recursive_tree(series, params=self.params)
    return self

  # Função recursiva para encontrar os nós e criar a árvore
  def _recursive_tree(self, series: Iterable, params: dict, curr_depth=0):
    stop_func_result, series = self.stop_func(series, curr_depth)
    if (stop_func_result >= 0) or (len(series) < (self.min_dist * 2)):
      return None
    # Achando a posição de corte e pegando os parâmetros da função de corte
    # Isso permite que a função de corte altere os parâmetros pra chamada dos próximos nós para otimizar os cortes
    cut_pos, params, heatmap = self._find_cut(series=series[1:-1], params=params, depth=curr_depth)
    # Soma um na posição porque passa a série sem o primeiro e último elemento porque não são opções de posição de corte
    cut_pos+=1
    # Retorna None se ele não achar corte válido, indicando que o nó é folha
    if cut_pos < 0:
      return None
    if list(heatmap):
      min_hm = min(heatmap)
      max_hm = max(heatmap)
      if min_hm != max_hm:
        heatmap = [(hm_val-min_hm)/(max_hm-min_hm) for hm_val in heatmap]
  
    node = TreeNode({'cut_pos': cut_pos, 'heatmap': heatmap, 'entropy': stop_func_result})
    node.left = self._recursive_tree(series[:cut_pos], params=params, curr_depth=curr_depth+1)
    node.right = self._recursive_tree(series[cut_pos:], params=params, curr_depth=curr_depth+1)

    # Retorna o nó
    return node

  def calculate_entropy_gain(self):
    self._rec_entropy_gain(self.tree.root, None)
    return self
  
  def _rec_entropy_gain(self, node: TreeNode, parent: TreeNode):
    if node is None:
      return
    if parent is None:
      node.cont['entropy_gain'] = 0
    else:
      node.cont['entropy_gain'] = parent.cont['entropy'] - node.cont['entropy']
    self._rec_entropy_gain(node.left, node)
    self._rec_entropy_gain(node.right, node)
  
  # def get_tree_info(self):
  #   self._rec_tree_info(self, self.tree.root)

  # def _rec_tree_info(self, node):
    
  
  # Função que encontra a posição de corte, única para cada método de corte
  def _find_cut(self, series: Iterable, params: dict, depth=0) -> Tuple[int, dict]:
    pass

  def apply_on_leaves(self, function, series: Iterable) -> List:
    return list(map(function, self.tree.cut_series(series)))

  def to_list(self) -> List:
    return self.tree.to_list()

  def get_heatmap(self) -> Tuple[List, List]:
    heatmap = XTSTree._get_heatmap(self.tree.root)
    if len(heatmap) == 0:
      return []
    return heatmap
    # return heatmap
  
  @staticmethod
  def _get_heatmap(node: TreeNode, par_heatmap: List=[]) -> Tuple[List, List]:
    if node == None:
      return par_heatmap

    if len(par_heatmap) == 0:
      par_heatmap = node.cont['heatmap']

    heatmap = node.cont['heatmap']
    l_heatmap = XTSTree._get_heatmap(node.left, heatmap[:node.cont['cut_pos']])
    r_heatmap = XTSTree._get_heatmap(node.right, par_heatmap[node.cont['cut_pos']:])

    return l_heatmap + r_heatmap
    

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

  def get_items_by_depth(self):
    return XTSTree._get_items_by_depth(node=self.tree.root, last_depth=0)
    
  def get_cuts_by_depth(self):
    return XTSTree._get_cuts_by_depth(node=self.tree.root, last_depth=0)
  
  def __repr__(self):
    return self.summary()
  
  def summary(self):
    return XTSTree._get_items_by_depth_side(self.tree.root, depth=0, side_prefix='Root')

  @staticmethod
  def _get_cuts_by_depth(node: TreeNode, last_depth: int):
    if node is None:
      return {}
    items_dict = {last_depth: [node.cont['cut_pos']]}
    for depth, val in XTSTree._get_cuts_by_depth(node=node.left, last_depth=last_depth + 1).items():
      if depth in items_dict:
        items_dict[depth] += val
      else:
        items_dict[depth] = val
    for depth, val in XTSTree._get_cuts_by_depth(node=node.right, last_depth=last_depth + 1).items():
      if depth in items_dict:
        items_dict[depth] += [item + node.cont['cut_pos'] for item in val]
      else:
        items_dict[depth] = [item + node.cont['cut_pos'] for item in val]
    
    return items_dict

  @staticmethod
  def _get_items_by_depth(node: TreeNode, last_depth: int):
    if node is None:
      return {}
    items_dict = {last_depth: [node.cont]}
    for depth, val in XTSTree._get_items_by_depth(node=node.left, last_depth=last_depth + 1).items():
      if depth in items_dict:
        items_dict[depth] += val
      else:
        items_dict[depth] = val
    for depth, val in XTSTree._get_items_by_depth(node=node.right, last_depth=last_depth + 1).items():
      # for item in val:
      #   item['cut_pos'] += node.cont['cut_pos']
      if depth in items_dict:
        items_dict[depth] += val
      else:
        items_dict[depth] = val
    
    return items_dict
  
  @staticmethod
  def _get_items_by_depth_side(node: TreeNode, depth: int, side_prefix='Root'):
    if node is None:
      return {}
    items_dict = {depth: [{side_prefix: node.cont['cut_pos']}]}
    if side_prefix == 'Root':
      side_prefix = ''
    for key, val in XTSTree._get_items_by_depth_side(node=node.left, depth=depth + 1, side_prefix=side_prefix+'L').items():
      if key in items_dict:
        items_dict[key] += val
      else:
        items_dict[key] = val
    for key, val in XTSTree._get_items_by_depth_side(node=node.right, depth=depth + 1, side_prefix=side_prefix+'R').items():
      if key in items_dict:
        items_dict[key] += [{side: cut+node.cont['cut_pos']} for item in val for side, cut in item.items()]
      else:
        items_dict[key] = [{side: cut+node.cont['cut_pos']} for item in val for side, cut in item.items()]
    
    return items_dict

  @staticmethod
  def _get_cuts(node: TreeNode, series: Iterable):
    if node is None:
      return [series]
    return [*XTSTree._get_cuts(node.left, series[:node.cont['cut_pos']]), *XTSTree._get_cuts(node.right, series[node.cont['cut_pos']:])]

  @staticmethod
  def _get_cut_points(node: TreeNode) -> List:
    if node is None:
      return []
    return [*XTSTree._get_cut_points(node.left), node.cont['cut_pos'], *[cut + node.cont['cut_pos'] for cut in XTSTree._get_cut_points(node.right)]]