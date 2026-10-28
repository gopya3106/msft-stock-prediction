import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

TARGET = "MSFT_next_week_ret"


def load_split(path, test_frac=0.2):
    """Chronological split: first 80% of weeks train, last 20% test. No shuffling."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    split = int(len(df) * (1 - test_frac))
    train, test = df.iloc[:split], df.iloc[split:]
    return (train.drop(columns=TARGET), test.drop(columns=TARGET),
            train[TARGET], test[TARGET])


def evaluate(y_true, y_pred, y_train_mean, name="model"):
    """Same metrics for every model."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # Out-of-sample R2 measured against the training-mean forecast
    r2_oos = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_train_mean) ** 2)
    dir_acc = np.mean((y_true > 0) == (y_pred > 0))
    return {"model": name, "RMSE": rmse, "MAE": mae, "R2": r2,
            "R2_OOS": r2_oos, "DirAcc": dir_acc}
