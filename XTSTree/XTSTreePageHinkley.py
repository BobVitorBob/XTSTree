from skmultiflow.drift_detection import PageHinkley
from XTSTree import XTSTree

class XTSTreePageHinkley(XTSTree):
  
  def __init__(self, stop_condition: str='depth', stop_val=2, max_iter=1000, min_dist=30, min_instances: int=30, delta: float=0.005, starting_threshold: float=50.0, alpha: float=1 - 0.0001):
    self.min_instances = min_instances
    self.delta = delta
    self.threshold = starting_threshold
    self.alpha = alpha
    super().__init__(stop_condition=stop_condition, stop_val=stop_val, max_iter=max_iter, min_dist=min_dist, params={'max_threshold': -1, 'threshold': self.threshold})

  def _find_cut(self, series: Iterable, params: dict):
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
      ph = PageHinkley(min_instances=self.min_dist, delta=self.delta, threshold=threshold)
      cut_pos = -1
      n_cuts = 0
      for i, val in enumerate(series[:-self.min_dist]):
        ph.add_element(val)
        if ph.detected_change():
          n_cuts += 1
          cut_pos = i
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
            break
      if n_cuts == 1:
        # Achou apenas um corte, o threshold máximo para as próximas iterações vira o threshold atual porque thresholds maiores não vão retornar cortes nas séries cortadas
        params['max_threshold'] = threshold
        params['threshold'] = threshold/2
        return cut_pos, params
      elif n_cuts < 1:
        # Se não achou corte, o threshold máximo vira o atual e faz a busca binária no threshold
        max_threshold = threshold
        threshold -= (threshold-min_threshold)/2
    
    # Se estourar o máximo de iterações, retorna que não achou um corte ali
    print(f'Não achei um corte, tamanho da série {len(series)}, {threshold}, {n_cuts}')
    return -1, params
