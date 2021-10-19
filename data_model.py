from datetime import datetime, timedelta

import numpy as np
from pygam import LinearGAM
from sklearn.linear_model import LinearRegression

from dataframe_formatting import (
    make_train_test_all,
    turn_array_into_dataframe,
)


def linear_reg_prediction(train_x, train_y, test_x, test_y):
    lin_reg = LinearRegression()
    # train_x = train_x.drop("ds", axis=1)
    # test_x = test_x.drop("ds", axis=1)
    lin_reg.fit(train_x, train_y)
    train_predictions = lin_reg.predict(train_x)
    test_predictions = lin_reg.predict(test_x)
    return lin_reg, train_predictions, test_predictions


def train_test_split(df, split_date="01-Jan-2020"):
    df_train = df.loc[df.index <= split_date].copy()
    df_test = df.loc[df.index > split_date].copy()
    return df_train, df_test


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def linear_gam(X_train, y_train):
    gam = LinearGAM(lam=10, n_splines=10, basis="cp").gridsearch(X_train, y_train)
    # gam = LinearGAM(basis="cp").gridsearch(X_train, y_train)
    # gam = GAM(
    #    s(0)
    #    + s(1)
    #    + s(2)
    #    + s(3)
    #    + s(4, basis="cp")
    #    + s(5, basis="cp")
    #    + s(6)
    #    + s(7)
    #    + s(8)
    #    + s(9)
    #    + s(10)
    #    + s(11)
    #    + s(12)
    #    + s(13)
    #    + s(14)
    #    + s(15, basis="cp")
    #    + s(16, basis="cp"),
    #    lam=30,
    #    n_splines=10,
    # ).gridsearch(X_train, y_train)
    return gam


def weekly_prediction(df, start_date, days, save=False):
    df_train, df_test = train_test_split(df, start_date)
    start_date = datetime.strptime(start_date, "%d-%m-%Y")
    end_date = timedelta(days)
    t = start_date + end_date
    new_start_date = start_date + timedelta(hours=19)

    df_test = df_test.loc[new_start_date:t].copy()
    X_train, y_train, _ = make_train_test_all(df_train)
    X_test, y_test, test_time = make_train_test_all(df_test)
    gam = linear_gam(X_train, y_train)
    y_prediction = gam.predict(X_test)
    arrays = [test_time, y_prediction]
    column_names = ["Date", "y_prediction"]
    df_prediction = turn_array_into_dataframe(arrays, column_names)
    file_name = (
        "/home/mehdi/time_series/prediction_" + str(new_start_date) + "to" + str(t)
    )
    print(file_name)
    if save:
        df_prediction.to_pickle(file_name)
    score = mean_absolute_percentage_error(y_test, y_prediction)
    return df_prediction, score, gam


def make_figure(df, start_date, hours=0, days=0, weeks=0, marker="o"):
    start_date = datetime.strptime(start_date, "%d-%m-%Y")
    end_date = (
        start_date
        + timedelta(hours=hours)
        + timedelta(days=days)
        + timedelta(weeks=weeks)
    )
    print(end_date)
    df = df.loc[start_date:end_date].copy()
    df.plot(figsize=(15, 5))
