import numpy as np


def mean_absolute_error(df_true, df_predict):
    relative = np.abs(df_true["y"] - df_predict["y"])
    return np.sum(relative)


def mean_squared_error(df_true, df_predict):
    relative = np.square(df_true["y"] - df_predict["y"])
    return np.sum(relative)


def rmse(df_true, df_predict):
    relative = np.square(df_true["y"] - df_predict["y"])
    return np.sqrt(np.sum(relative) / len(df_true))


def mean_absolute_percentage_error(df_true, df_predict):
    relative = np.abs(df_true["y"] - df_predict["y"]) / df_true["y"]
    return 1 / len(df_true) * np.sum(relative) * 100
