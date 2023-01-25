from XTSTree.XTSTree import XTSTree
from collections.abc import Iterable
from river.drift import KSWIN

class XTSTreeKSWIN(XTSTree):
  
  def __init__(self,
              # Parâmetros gerais
               stop_condition: str='depth',
               stop_val=2,
               max_iter=1000,
               min_dist:int=0,
              # Parâmetros KSWIN
               min_alpha:float=0,
               max_alpha:float=None,
               min_window_size:int=0,
               max_window_size:int=100,
               min_stat_size:int=0,
               max_stat_size:int=30,
               stat_step:float=0.1,
               window_step:float=0.1,
               seed:int=None,
               window:Iterable=None,
            ):
    self.min_alpha = min_alpha
    self.max_alpha = max_alpha
    self.min_window_size = min_window_size
    self.max_window_size = max_window_size
    self.min_stat_size = min_stat_size
    self.max_stat_size = max_stat_size
    self.stat_step = stat_step
    self.window_step = window_step
    self.seed = seed
    self.window = window
    super().__init__(
      stop_condition=stop_condition,
      stop_val=stop_val,
      max_iter=max_iter,
      min_dist=min_dist,
      params=None,
    )

  def _find_cut(self, series: Iterable, params: dict, depth=0):
    last_cuts_params = {}
    # Iterando pelos tamanhos de janela possíveis
    for window_size in np.arange(self.min_window_size, self.max_window_size, self.window_step):
      window_size = int(window_size)
      for stat_size in np.arange(self.min_stat_size, self.max_stat_size, self.stat_step):
        stat_size = int(stat_size)
        alpha = self.min_alpha
        min_alpha = self.max_alpha
        max_alpha = self.min_alpha
        for n_iter_a in range(self.max_iter):
          kswin = KSWIN(alpha=alpha, window_size=window_size, stat_size=stat_size, window=self.window, seed=self.seed)
          cut_pos = []
          for i, val in enumerate(series[:-self.min_dist]):
            kswin.update(val)
            if kswin.drift_detected:
              cut_pos.append(i)
              # Se detectou mais de um corte, então tem que diminuir o alpha
              if len(cut_pos) > 1:
                last_cuts_params['alpha'] = alpha
                last_cuts_params['window_size'] = window_size
                last_cuts_params['stat_size'] = stat_size
                max_alpha = alpha
                alpha -= (alpha-min_alpha)/2
                break
          n_cuts = len(cut_pos)
          if n_cuts == 1:
            # Achou só um corte, não altera os máximos e mínimos porque tem mais de um nível
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
      # Se não achou cortes, pega os parâmetros que acharam cortes da última vez
      kswin = KSWIN(alpha=last_cuts_params['min_alpha'], window_size=self.last_cuts_params['window_size'], window=self.window, stat_size=self.last_cuts_params['stat_size'], seed=self.seed)
      cut_pos = []
      for i, val in enumerate(series[:-self.min_dist]):
        kswin.update(val)
        if kswin.drift_detected:
          cut_pos.append(i)
    
    if len(cut_pos) == 0:
      print(f'Não achei nenhum corte em {self.max_iter} iterações, nó tem que ser folha')
      return -1, params
    print(f'Não achei só um corte, escolhendo corte que gera maior pontuação, {len(series)}, {threshold}, {n_cuts}')
    max_stat = -1
    final_cut = -1
    for pos in cut_pos:
      pos_stat = self.stop_func(series[:pos]) + self.stop_func(series[pos:])
      if pos_stat > max_stat:
        max_stat = pos_stat
        final_cut = pos
    return final_cut, params