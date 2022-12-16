from XTSTree.XTSTree import XTSTree
from collections.abc import Iterable
from river.drift import KSWIN

class XTSTreeKSWIN(XTSTree):
  
  def __init__(self,
              # Parâmetros gerais
               stop_condition: str='depth',
               stop_val=2,
               max_iter=1000,
               min_dist:int=30,
              # Parâmetros KSWIN
               starting_alpha:float=0.005,
               window_size:int=100,
               stat_size:int=30,
               seed:int=None,
               window:Iterable=None,
            ):
    self.starting_alpha = starting_alpha
    self.window_size = window_size
    self.stat_size = stat_size
    self.seed = seed
    self.window = window
    super().__init__(stop_condition=stop_condition, stop_val=stop_val, max_iter=max_iter, min_dist=min_dist, params={'max_alpha': -1, 'alpha': self.starting_alpha})

  def _find_cut(self, series: Iterable, params: dict):
    # Faz uma cópia dos parâmetros porque vai retornar uma cópia dos parâmetros alterados
    # A cópia é feita dentro da função de corte porque pode não precisar alterar os parâmetros dependendo do método de corte
    params = dict(params)
    alpha = params['alpha']
    min_alpha = 0
    max_alpha = params['max_alpha']
    # Limitando o número de iterações pra um máximo
    for n_iter in range(self.max_iter):
      # Pra reforçar a distância mínima entre cortes, o número de instâncias mínimas até detectar mudança é colocado como a distância mínima, e a série é analisada até os último min_dist elementos.
      kswin = KSWIN(alpha=alpha, window_size=self.window_size, window=self.window, stat_size=self.stat_size, seed=self.seed)
      cut_pos = []
      for i, val in enumerate(series[:-self.min_dist]):
        kswin.update(val)
        if kswin.drift_detected:
          cut_pos.append(i)
          # Se detectou mais de um corte, então tem que diminuir o alpha
          if len(cut_pos) > 1:
            max_alpha = alpha
            alpha -= (alpha-min_alpha)/2
            break
      n_cuts = len(cut_pos)
      if n_cuts == 1:
        # Achou apenas um corte, o alpha máximo para as próximas iterações vira o alpha atual porque alphas maiores não vão retornar cortes nas séries cortadas
        params['max_alpha'] = alpha
        params['alpha'] = alpha/2
        return cut_pos[0], params
      elif n_cuts < 1:
        # Se não achou corte, tem que aumentar o alpha
        # Atualiza o alpha mínimo
        min_alpha = alpha
        # E faz a busca binária do alpha
        # Se for menor que 0 é porque não foi definido, então aumenta o alpha em 50%
        if max_alpha < 0:
          alpha += alpha/2
        else:
          alpha += (max_alpha - alpha)/2
    

    if n_cuts == 0:
      # Se não achou cortes, pega os cortes da alpha máxima.
      kswin = KSWIN(alpha=min_alpha, window_size=self.window_size, window=self.window, stat_size=self.stat_size, seed=self.seed)
      cut_pos = []
      for i, val in enumerate(series[:-self.min_dist]):
        kswin.update(val)
        if kswin.drift_detected:
          cut_pos.append(i)
    

    # Se estourar o máximo de iterações, escolhe o ponto que gera mais estacionariedade
    print(f'Não achei um corte, pegando melhor {len(series)}, {alpha}, {n_cuts}')
    max_stat = -1
    final_cut = -1
    for pos in cut_pos:
      pos_stat = self.stop_func(series[:pos]) + self.stop_func(series[pos:])
      if pos_stat > max_stat:
        max_stat = pos_stat
        final_cut = pos
    params['max_alpha'] = max_alpha
    params['alpha'] = alpha
    return final_cut, params