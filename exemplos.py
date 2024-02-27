import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation
import numpy as np
from pysr import PySRRegressor
from statsmodels.tsa.stattools import adfuller

def get_regressor():
    return PySRRegressor(
        binary_operators=['+', '-', '*', '/'],
        unary_operators=['sin'],
        niterations=5,
        populations=10,
        population_size=20,
        progress=False,
        model_selection='best',
        equation_file=f'./symbreg_objects/test.csv',
        verbosity=0,
        temp_equation_file=False
    )

def instancia(prev, root, mean=0, var=1):
  return root*prev + np.random.normal(loc=mean, scale=var)


est_series = []
nest_series = [0]
n_instances = 500

def series_formula(index):
  return (index/100) * np.sin(index/50) + index/50

for i in range(n_instances):
  est_series.append(series_formula(i))

model = get_regressor()
X = [[i] for i in range(n_instances)]
model.fit(X, est_series, variable_names=['index'])
yhat = model.predict(X)

# for i in range(n_instances):
#   est_series.append(instancia(est_series[i], root=0.8))
  
# for i in range(n_instances):
#   nest_series.append(instancia(nest_series[i], root=1))
  
f = LogFormatterSciNotation(base=10, minor_thresholds=(-10, 10))
  
# fig, ((axEST, axnEST)) = plt.subplots(nrows=2, ncols=1, figsize=(8,5))
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))

ax.plot(est_series, linewidth=0.5)
ax.plot(yhat, linewidth=0.5)
ax.set_title(f'Formula: {model.get_best()["equation"]}')
# ax.set_ylim((min([*nest_series, *est_series]) + 2, max([*nest_series, *est_series]) + 2))

# axEST.plot(est_series, linewidth=0.5)
# axEST.set_title(f'ADF test value: {f.format_data(adfuller(est_series, regression="c")[1])} (< 0.05 means stationarity)')
# axEST.set_ylim((min([*nest_series, *est_series]) + 2, max([*nest_series, *est_series]) + 2))
# axnEST.plot(nest_series, linewidth=0.5)
# axnEST.set_title(f'ADF test value: {f.format_data(round(adfuller(nest_series, regression="c")[1], 3))} (< 0.05 means stationarity)')
# axEST.set_ylim((min([*nest_series, *est_series]) + 2, max([*nest_series, *est_series]) + 2))

plt.subplots_adjust(hspace=0.3)
plt.savefig(f'pysr.pdf', bbox_inches='tight')
plt.show()
