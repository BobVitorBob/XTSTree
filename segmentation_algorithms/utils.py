from sklearn.linear_model import LinearRegression
import numpy as np

def mae(y, y_hat):
    return np.mean(np.abs(y - y_hat))

def rmse(y, y_hat):
    return np.sqrt(np.mean(np.square(y - y_hat)))

def group_data(data, window_size=96):
  """
  Separa o vetor data em pares de vetores X de entrada e valor de predição y
  data: Conjunto de dados
  window_size: Tamanho dos vetores de dados em X
  Retorna o vetor X e Y
  """
  data_X = []
  data_Y = []
  len_data = len(data)
  for i in range(len_data - window_size):
      data_X.append(data[i:i + window_size])
      data_Y.append(data[i + window_size])
  return np.array(data_X), np.array(data_Y)

def create_series():
  generator = np.random.default_rng(42)
  series = np.concatenate([
    np.array(generator.uniform(-10, 10, 100)) + np.linspace(0, 200, 100)[::-1],
    
    np.array(generator.uniform(-10, 10, 200)),
    np.array(generator.uniform(-10, 10, 150)) + np.linspace(0, 200, 150),
    np.array(generator.uniform(-10, 10, 50)) + np.linspace(0, 200, 50)[::-1],

    np.array(generator.uniform(-10, 10, 200)),
    np.array(generator.uniform(-10, 10, 150)) + np.linspace(0, 200, 150),
    np.array(generator.uniform(-10, 10, 50)) + np.linspace(0, 200, 50)[::-1],

    np.array(generator.uniform(-10, 10, 200)),
    np.array(generator.uniform(-10, 10, 150)) + np.linspace(0, 200, 150),
    np.array(generator.uniform(-10, 10, 50)) + np.linspace(0, 200, 50)[::-1],
  ])
  series = series + (np.sin([(i/20) for i in range(len(series))])*60)
  return series

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