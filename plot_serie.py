import pandas as pd
from plot import plot

df = pd.read_csv(f'./datasets/base datasets/export_automaticas_23025122_umidrelmed2m.csv')
df = df[df.columns[-1]]

plot(list(df)[:96*20])