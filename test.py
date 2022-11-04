from Separators.SeparatorPageHinkley import SeparatorPageHinkley
from plot import plot
import math
import random

if __name__ == '__main__':
	sep = SeparatorPageHinkley()
	series = [(math.sin(x/100)) for x in range(1000)] + [(math.sin(x/10)) for x in range(1000)]
	plot(series)
	cuts = sep.create_splits(series)
	print(cuts)
	plot(series, divisions=cuts)