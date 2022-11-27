import pandas as pd
import numpy as np


def mae(y, y_hat):
    return np.mean(np.abs(y - y_hat))


def mse(y, y_hat):
    return np.mean(np.square(y - y_hat))


def rmse(y, y_hat):
    return np.sqrt(np.mean(np.square(y - y_hat)))


def mape(y, y_hat):
    return np.mean(np.abs((y - y_hat) / y) * 100)


# Median Relative Absolute Error (MdRAE)
# If our model’s forecast equals to the benchmark’s forecast then the result is 1.
# If the benchmarks forecast are better than ours then the result will be above > 1.
# If ours is better than it’s below 1.
def mdrae(y, y_hat, bnchmrk):
    return np.median(np.abs(y - y_hat) / np.abs(y - bnchmrk))


# Geometric Mean Relative Absolute Error (GMRAE)
def gmrae(y, y_hat, bnchmrk):
    abs_scaled_errors = np.abs(y - y_hat) / np.abs(y - bnchmrk)
    return np.exp(np.mean(np.log(abs_scaled_errors)))


# Mean Absolute Scaled Error (MASE)
def mase(y, y_hat, y_train):
    naive_y_hat = y_train[:-1]
    naive_y = y_train[1:]
    mae_in_sample = np.mean(np.abs(naive_y - naive_y_hat))
    mae = np.mean(np.abs(y - y_hat))

    return mae / mae_in_sample


path = "results/results_arima_umidrelmed2m.csv"

df = pd.read_csv(path)

df["diff_original"] = df["series"] - df["pred_full"]
df["diff_xtstree"] = df["series"] - df["pred_cuts"]

print("-" * 60)
print("Absolute Accumulated Error:")
print("Original:", df["diff_original"].abs().sum().round(2))
print("XTSTree:", df["diff_xtstree"].abs().sum().round(2))

metrics = [mae, mse, mape, mse, rmse]

for metric in metrics:
    print("-" * 60)
    print(metric.__name__)
    print("Original:", metric(df["series"], df["pred_full"]).round(4))
    print("XTSTree:", metric(df["series"], df["pred_cuts"]).round(4))

metrics = [mdrae, gmrae]
for metric in metrics:
    print("-" * 60)
    print(metric.__name__)
    print("Result:", metric(df["series"], df["pred_cuts"], df["pred_full"]).round(4))
