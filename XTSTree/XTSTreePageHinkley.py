from XTSTree.XTSTree import XTSTree
from collections.abc import Iterable
from river.drift import PageHinkley

class XTSTreePageHinkley(XTSTree):
  
  def __init__(self, stop_condition: str='depth', stop_val=2, max_iter=1000, min_dist:int=0, min_instances: int=30, delta: float=0.005, starting_threshold: float=50.0, alpha: float=1 - 0.0001):
    self.min_instances = min_instances
    self.delta = delta
    self.threshold = starting_threshold
    self.alpha = alpha
    super().__init__(stop_condition=stop_condition, stop_val=stop_val, max_iter=max_iter, min_dist=min_dist, params={'max_threshold': -1, 'threshold': self.threshold})

  def _find_cut(self, series: Iterable, params: dict, depth=0):
    # Faz uma cópia dos parâmetros porque vai retornar uma cópia dos parâmetros alterados
    # A cópia é feita dentro da função de corte porque pode não precisar alterar os parâmetros dependendo do método de corte
    params = dict(params)
    # Threshold mínimo sempre é 0, máximo deve começar com -1 mas idealmente é alterado para otimizar a busca
    threshold = params['threshold']
    min_threshold = 0
    max_threshold = params['max_threshold']
    # Limitando o número de iterações pra um máximo
    for n_iter in range(self.max_iter):
      # Pra reforçar a distância mínima entre cortes, o número de instâncias mínimas até detectar mudança é colocado como a distância mínima, e a série é analisada até os último min_dist elementos.
      ph = PageHinkley(min_instances=self.min_instances, delta=self.delta, threshold=threshold)
      heat_map_increase = []
      heat_map_decrease = []
      cut_pos = []
      # Rever funcionamento do min_dist junto dos heatmaps
      for i, val in enumerate(series):
        ph.update(val)
        heat_map_increase_step = ph._sum_increase - ph._min_increase
        heat_map_decrease_step = ph._max_decrease - ph._sum_decrease
        heat_map_increase.append(threshold - heat_map_increase_step)
        heat_map_decrease.append(threshold - heat_map_decrease_step)
        if ph.drift_detected:
          cut_pos.append(i)
      n_cuts = len(cut_pos)
      # Se detectou mais de um corte, então tem que aumentar o threshold
      if n_cuts > 1:
        # Atualiza o threshold mínimo
        min_threshold = threshold
        # E faz a busca binária do threshold
        # Se for menor que 0 é porque não foi definido, então aumenta o threshold em 50%
        if max_threshold < 0:
          threshold += threshold/2
        else:
          threshold += (max_threshold - threshold)/2
      elif n_cuts == 1:
        # Achou apenas um corte, o threshold máximo para as próximas iterações vira o threshold atual porque thresholds maiores não vão retornar cortes nas séries cortadas
        params['max_threshold'] = threshold
        params['threshold'] = threshold/2
        return cut_pos[0], params, [min(hm_inc, hm_dec) for hm_inc, hm_dec in zip(heat_map_increase, heat_map_decrease)]
      elif n_cuts < 1:
        # Se não achou corte, o threshold máximo vira o atual e faz a busca binária no threshold
        max_threshold = threshold
        threshold -= (threshold-min_threshold)/2
    

    if n_cuts == 0:
      # Se não achou cortes, pega os cortes da threshold máxima.
      ph = PageHinkley(min_instances=self.min_instances, delta=self.delta, threshold=min_threshold)
      heat_map_increase = []
      heat_map_decrease = []
      cut_pos = []
      for i, val in enumerate(series):
        ph.update(val)
        heat_map_increase_step = ph._sum_increase - ph._min_increase
        heat_map_decrease_step = ph._max_decrease - ph._sum_decrease
        heat_map_increase.append(min_threshold - heat_map_increase_step)
        heat_map_decrease.append(min_threshold - heat_map_decrease_step)
        if ph.drift_detected:
          cut_pos.append(i)
    
    # Se estourar o máximo de iterações, escolhe o ponto que gera mais estacionariedade
    if len(cut_pos) == 0:
      print(f'Não achei nenhum corte em {self.max_iter} iterações, nó tem que ser folha')
      return -1, params, [min(hm_inc, hm_dec) for hm_inc, hm_dec in zip(heat_map_increase, heat_map_decrease)]
    print(f'Não achei só um corte, escolhendo corte que gera maior pontuação, {len(series)}, {threshold}, {n_cuts}')
    sf1, _ = self.stop_func(series[:cut_pos[0]], depth)
    sf2, _ = self.stop_func(series[cut_pos[0]:], depth)
    max_stat = sf1 + sf2
    final_cut = cut_pos[0]
    for pos in cut_pos[1:]:
      sf1, _ = self.stop_func(series[:pos], depth)
      sf2, _ = self.stop_func(series[pos:], depth)
      pos_stat = sf1 + sf2
      if pos_stat > max_stat:
        max_stat = pos_stat
        final_cut = pos
    params['max_threshold'] = max_threshold
    params['threshold'] = threshold
    return final_cut, params, [min(hm_inc, hm_dec) for hm_inc, hm_dec in zip(heat_map_increase, heat_map_decrease)]