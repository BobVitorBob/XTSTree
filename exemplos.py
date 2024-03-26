import matplotlib.pyplot as plt
from XTSTree.XTSTreePageHinkley import XTSTreePageHinkley
from plot import plot
import pandas as pd
from matplotlib.ticker import ScalarFormatter
import numpy as np
from pysr import PySRRegressor
from statsmodels.tsa.stattools import adfuller

# def get_regressor():
#     return PySRRegressor(
#         binary_operators=['+', '-', '*', '/'],
#         unary_operators=['sin'],
#         niterations=5,
#         populations=10,
#         population_size=20,
#         progress=False,
#         model_selection='best',
#         equation_file=f'./symbreg_objects/test.csv',
#         verbosity=0,
#         temp_equation_file=False
#     )

# def instancia(prev, root, mean=0, var=1):
#   return root*prev + np.random.normal(loc=mean, scale=var)


# est_series = [0]
# nest_series = [0]
# n_instances = 500

# # def series_formula(index):
# #   return (index/100) * np.sin(index/50) + index/50

# # for i in range(n_instances):
# #   est_series.append(series_formula(i))

# # model = get_regressor()
# # X = [[i] for i in range(n_instances)]
# # model.fit(X, est_series, variable_names=['index'])
# # yhat = model.predict(X)

# for i in range(n_instances):
#   est_series.append(instancia(est_series[i], root=0.8))
  
# for i in range(n_instances):
#   nest_series.append(instancia(nest_series[i], root=1))
  
# f = ScalarFormatter(
#   useMathText=True,
#   # base=10, minor_thresholds=(-10, 10)
# )
  
# f.set_scientific(True)
# fig, ((axEST, axnEST)) = plt.subplots(nrows=2, ncols=1, figsize=(8,5))
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))

# ax.plot(est_series, linewidth=0.5)
# ax.plot(yhat, linewidth=0.5)
# ax.set_title(f'Formula: {model.get_best()["equation"]}')
# ax.set_ylim((min([*nest_series, *est_series]) + 2, max([*nest_series, *est_series]) + 2))

# axEST.plot(est_series, linewidth=0.5)
# axEST.set_title(f'ADF test value: ${f.format_data(round(adfuller(est_series, regression="c")[1], 12))}$ (< 0.05 means stationarity)')
# axEST.set_ylim((min([*nest_series, *est_series]) + 2, max([*nest_series, *est_series]) + 2))
# axnEST.plot(nest_series, linewidth=0.5)
# axnEST.set_title(f'ADF test value: ${f.format_data(round(adfuller(nest_series, regression="c")[1], 3))}$ (< 0.05 means stationarity)')
# axEST.set_ylim((min([*nest_series, *est_series]) + 2, max([*nest_series, *est_series]) + 2))

# plt.subplots_adjust(hspace=0.5)
# plt.savefig(f'comp_est_nest.pdf', bbox_inches='tight')
# plt.show()

# ex1
file = '5dias_umidrelmed2m_2019-11-01 _ 2019-11-06.csv'
df = pd.read_csv(f'./datasets/umidrelmed2m/5dias/{file}')
series1 = df[df.columns[-1]]
plot(series1, save=True, img_name='xtstree_example_base_series_1.pdf', max_y=120, min_y=0, show=False)
# ex2
file = '5dias_umidrelmed2m_2018-02-23 _ 2018-02-28.csv'
df = pd.read_csv(f'./datasets/umidrelmed2m/5dias/{file}')
series2 = df[df.columns[-1]]
plot(series2, save=True, img_name='xtstree_example_base_series_2.pdf', max_y=120, min_y=0, show=False)
# ex3
file = '5dias_umidrelmed2m_2019-05-10 _ 2019-05-17.csv'
df = pd.read_csv(f'./datasets/umidrelmed2m/5dias/{file}')
series3 = df[df.columns[-1]]
plot(series3, save=True, img_name='xtstree_example_base_series_3.pdf', max_y=120, min_y=0, show=False)

model1 = XTSTreePageHinkley(stop_condition='adf', stop_val=0, max_iter=100, min_dist=30,)
model2 = XTSTreePageHinkley(stop_condition='adf', stop_val=0, max_iter=100, min_dist=30,)
model3 = XTSTreePageHinkley(stop_condition='adf', stop_val=0, max_iter=100, min_dist=30,)

model1 = model1.create_splits(series1)
model2 = model2.create_splits(series2)
model3 = model3.create_splits(series3)

mean_by_cut, n_items, tot_depth = model1.calc_mean_entropy_gain_by_cut()
mean_by_cut, n_items, tot_depth = model2.calc_mean_entropy_gain_by_cut()
mean_by_cut, n_items, tot_depth = model3.calc_mean_entropy_gain_by_cut()

segments1 = model1.cut_points()
segments2 = model2.cut_points()
segments3 = model3.cut_points()

print(model1.summary())
print(model2.summary())
print(model3.summary())

plot(series1, divisions=segments1, save=True, img_name='xtstree_example_1.pdf', max_y=120, min_y=0, show=False)
plot(series2, divisions=segments2, save=True, img_name='xtstree_example_2.pdf', max_y=120, min_y=0, show=False)
plot(series3, divisions=segments3, save=True, img_name='xtstree_example_3.pdf', max_y=120, min_y=0, show=False)