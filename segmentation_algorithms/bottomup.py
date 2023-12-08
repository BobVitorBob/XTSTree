# %%
from sklearn.linear_model import LinearRegression
import numpy as np
from plot import plot
from sklearn.linear_model import LinearRegression

def mae(y, y_hat):
  return np.mean(np.abs(np.array(y) - np.array(y_hat)))

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

def lr_error(X, y, model):
  model, _, _, _ = apply_lr(X, y)
  error = mae(y, model.predict(X))
  return error, model

# full_error, original_model  = apply_lr(np.array(range(len(series)), series))
class BottomUp:
  
  def __init__(self, series):
    self.series = series
    self._reset()
  
  def _reset(self, init_seg_size=2):
    self.segments = []
    for i in range(0, len(self.series), init_seg_size):
      seg = self.series[i:i+init_seg_size]
      model, _, coef, intercept = apply_lr(list(range(i, i+init_seg_size)), seg)
      error = mae(seg, model.predict(np.array(range(i, i+init_seg_size)).reshape(-1, 1)))
      self.segments.append({'model': model, 'start': i, 'end': i+init_seg_size-1, 'seg': seg, 'error': error})

  def fit(self, threshold, mode='seg', init_seg_size=2, print_when=[], path='./'):
    self._reset(init_seg_size)
    curr_error = 0
    # error_threshold = full_error * 0.05
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
      curr_error = mae(self.series, predicted_series)
      if len(self.segments) in print_when:
        divs = [seg['end'] for seg in self.segments][:-1]
        plot(self.series,
          divisions=divs,
          show=False,
          save=True,
          show_axis=(False, False),
          figsize=(3.5,3),
	        dpi=720,
          img_name=f'{path}/{len(self.segments)}_segments.jpeg',
          # backgrounds=[
          # 	{'start': 0,'end': divs[0], 'color': '#0066FF'},
          # 	{'start': divs[0],'end': divs[1], 'color': '#D3321D'},
          # 	{'start': divs[1],'end': len(con_drift), 'color': '#66DF4D'}
          # ],
        )
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


