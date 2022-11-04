from Separators.SeparatorPageHinkley import SeparatorPageHinkley
from plot import plot
import math
import pandas as pd
import random

if __name__ == '__main__':
	sep = SeparatorPageHinkley()
	series = pd.read_csv('../../../Dados/por estacao/23025122/export_automaticas_23025122_umidrelmed2m.csv', nrows=2000)['umidrelmed2m']
	# series = [(math.sin(x/100)) for x in range(1000)] + [(math.sin(x/10)) for x in range(1000)]
	plot(series)
	cuts, threshold = sep.create_splits(series)
	print(cuts, threshold)
	plot(series, divisions=cuts, title=threshold)