# %%
from sklearn.linear_model import LinearRegression
import numpy as np
from plot import plot
from utils import *

# full_error, original_model  = apply_lr(np.array(range(len(series)), series))
class BottomUp:
  
  def __init__(self, series):
    self.series = series
    self._reset()
  
  def _reset(self):
    self.segments = []
    for i in range(0, len(self.series), 2):
      seg = self.series[i:i+2]
      model, _, coef, intercept = apply_lr(list(range(i, i+2)), seg)
      error = mae(seg, model.predict(np.array(range(i, i+2)).reshape(-1, 1)))
      self.segments.append({'model': model, 'start': i, 'end': i+1, 'seg': seg, 'error': error})

  def fit(threshold, mode='seg'):
    self.reset()
    curr_error = 0
    error_threshold = full_error * 0.05
    # while curr_error < error_threshold:
    while len(self.segments) > threshold:
      errors = []
      for i in range(len(self.segments)-1):
        seg1 = self.segments[i]
        seg2 = self.segments[i+1]
        combined_segs = [*seg1['seg'], *seg2['seg']]
        m_model, _, m_coef, m_intercept = apply_lr(np.array(range(seg1['start'], seg2['end']+1)), combined_segs)

        m_error = mae(combined_segs, m_model.predict(np.array(range(seg1['start'], seg2['end']+1)).reshape(-1, 1)))

        diff = m_error - (seg1['error'] + seg2['error'])/2

        errors.append(
          {'model': m_model, 'start': seg1['start'], 'end': seg2['end'], 'seg': combined_segs, 'error': m_error, 'diff': diff})

      min_loss = np.argmin([e['diff'] for e in errors])

      self.segments.pop(min_loss)
      self.segments.pop(min_loss)
      self.segments.insert(min_loss, errors[min_loss])
      predicted_series = []
      for seg in self.segments:
        prediction = seg['model'].predict(np.array(range(seg['start'], seg['end']+1)).reshape(-1, 1))
        predicted_series = [*predicted_series, *prediction]
      curr_error = mae(series, predicted_series)
    return self.segments
# %%
# print('erro sem segmentação', full_error)
# print('erro segmentado antes', erro_seg_inicio)
# print('erro segmentado depois', sum([seg['error'] for seg in segments]))
# print('Numero de segmentos: ', len(segments))

# %%
# predicted_series = []
# for seg in segments:
#   prediction = seg['model'].predict(np.array(range(seg['start'], seg['end']+1)).reshape(-1, 1))
#   predicted_series = [*predicted_series, *prediction]
#   # plot(seg['seg'], sec_plots=[prediction], title=e)
# error = mae(series, predicted_series)

# # %%
# plot(series, sec_plots=[predicted_series], title=error)
# plot(series, sec_plots=[original_model.predict(np.array(range(len(series))).reshape(-1, 1))], title=full_error)


