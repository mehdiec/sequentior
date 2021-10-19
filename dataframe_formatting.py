from datetime import datetime

import numpy as np
import pandas as pd


def read_RTE_to_dataframe(path, usecols, drop_NA=True):
    df = pd.read_excel(path, usecols=usecols)
    if drop_NA:
        df = df[df["Date"].notna()]
        df = df[df["Heures"].notna()]
    df.loc[:, "Date"] = pd.to_datetime(
        df.Date.astype(str) + " " + df.Heures.astype(str)
    )
    del df["Heures"]
    df.columns = ["ds", "y"]
    df = df.set_index("ds")
    df = df[df["y"] != "ND"]
    df = df[df["y"] > 100]
    df = df.dropna()
    return df


def read_climate_to_dataframe(path, selected_colulns=["DATE", "TMP", "DEW"]):
    df = pd.read_csv(path)
    dew = [
        float((x.replace(",", ".")).replace("-", "+")) / 10 for x in df["DEW"].values
    ]
    for i in range(len(dew)):
        if dew[i] > 211:
            dew[i] = dew[i - 1]
    df["DEW"] = dew
    temp = [float(x.replace(",", ".")) / 10 for x in df["TMP"].values]
    for i in range(len(temp)):
        if temp[i] > 44:
            temp[i] = temp[i - 1]
    df["TMP"] = temp
    T = [datetime.fromisoformat(x.replace("T", " ")) for x in df["DATE"].values]
    df["DATE"] = T
    df = df[selected_colulns]
    df = df.set_index("DATE")
    return df


def dataframe_model(path_conso, path_recording):
    df_conso = read_RTE_to_dataframe(path_conso, "C:E")
    df_recording = read_climate_to_dataframe(path_recording)
    df_train = pd.merge(df_conso, df_recording, left_index=True, right_index=True)
    return df_train


def join_dataframes(df1, df2):

    df = pd.merge(df1, df2, left_index=True, right_index=True)
    return df


def date_into_durations(df):
    df = df.copy()

    df["date_delta"] = (df["index"] - df["index"].min()) / np.timedelta64(1, "D")
    return df


def make_train_test_all(df_train_test):

    X_int = create_features(df_train_test, label="y")

    time_stamp = df_train_test.index.values
    df_train_test = df_train_test.reset_index()
    df = date_into_durations(df_train_test)
    evm_temperature = X_int["TMP"].ewm(span=10).mean()
    y = X_int["y"]
    conso_shift = y.shift(24)
    # for i in range(24):
    #    conso_shift.iloc[i] = y.iloc[i]
    T2 = 365
    cos_year = np.cos(2 * np.pi * df["date_delta"].values / T2)
    cos2_year = np.cos(4 * np.pi * df["date_delta"].values / T2)
    sin_year = np.sin(2 * np.pi * df["date_delta"].values / T2)
    sin2_year = np.sin(4 * np.pi * df["date_delta"].values / T2)
    T3 = 365 / 90
    cos_quater = np.cos(2 * np.pi * df["date_delta"].values / T3)
    cos2_quater = np.cos(4 * np.pi * df["date_delta"].values / T3)
    sin_quater = np.sin(2 * np.pi * df["date_delta"].values / T3)
    sin2_quater = np.sin(4 * np.pi * df["date_delta"].values / T3)

    X_fourier = np.array(
        [
            cos_year,
            cos2_year,
            sin_year,
            sin2_year,
            cos_quater,
            cos2_quater,
            sin_quater,
            sin2_quater,
            conso_shift,
            evm_temperature,
        ]
    )
    X_fourier = np.array([conso_shift, evm_temperature])
    # y = df["y"].values
    column_names = [
        "cos_year",
        "cos2_year",
        "sin_year",
        "sin2_year",
        "cos_quater",
        "cos2_quater",
        "sin_quater",
        "sin2_quater",
        "conso_shift",
        "evm_temperature",
    ]
    column_names = [
        "conso_shift",
        "evm_temperature",
    ]
    arrays = X_fourier
    df_fourier = turn_array_into_dataframe(arrays, column_names, time_stamp)
    # selected_column = [    "ds", "hour", "weekofyear", "DEW",    ]
    # X_int = X_int[["dayofweek", "hour"]]
    X = pd.concat([X_int, df_fourier], axis=1, sort=False)
    X = X.dropna()
    y = X["y"]
    time_list = X["ds"]
    X = X.drop("ds", 1)
    X = X.drop("y", 1)
    return X.values, y, time_list


def create_features(df, harmonic=1, label=None):
    """
    Creates time series features from datetime index.
    """
    T2 = 365 * 24 * 3600
    T3 = 3600 * 7 * 24
    T4 = 3600 * 24
    df = df.copy()
    df["ds"] = df.index
    df["timestamp"] = (df.index - pd.Timestamp("2007-01-01")) / np.timedelta64(1, "s")
    df["time"] = df["timestamp"] / (T2 * 10)

    df["pos_week"] = df["timestamp"] / T3 % 1
    df["pos_year"] = df["timestamp"] / T2 % 1

    df["pos_day"] = df["timestamp"] / T4 % 1
    for i in range(harmonic):
        i += 1
        df[f"pos_cos_year_{i}"] = np.cos(2 * i * np.pi * df["pos_year"])
        df[f"pos_sin_year_{i}"] = np.sin(2 * i * np.pi * df["pos_year"])
        df[f"pos_cos_week_{i}"] = np.cos(2 * i * np.pi * df["pos_week"])
        df[f"pos_sin_week_{i}"] = np.sin(2 * i * np.pi * df["pos_week"])
        df[f"pos_cos_day_{i}"] = np.cos(2 * i * np.pi * df["pos_day"])
        df[f"pos_sin_day_{i}"] = np.sin(2 * i * np.pi * df["pos_day"])
    for feature in label:
        if harmonic == 0:

            X = df[["time", "pos_year", "pos_week", "pos_day", "TMP", f"{feature}"]]

        else:
            list_harmonic = []
            for i in range(harmonic):

                i += 1
                list_harmonic += [
                    f"pos_cos_year_{i}",
                    f"pos_sin_year_{i}",
                    f"pos_cos_week_{i}",
                    f"pos_sin_week_{i}",
                    f"pos_cos_day_{i}",
                    f"pos_sin_day_{i}",
                ]
            full_columns = list_harmonic + [
                "time",
                "pos_year",
                "pos_week",
                "pos_day",
                "TMP",
                f"pos_cos_year_{i}",
                f"pos_sin_year_{i}",
                f"pos_cos_week_{i}",
                f"pos_sin_week_{i}",
                f"pos_cos_day_{i}",
                f"pos_sin_day_{i}",
                f"{feature}",
            ]
            X = df[full_columns]

    return X


def turn_array_into_dataframe(arrays, column_names, index=None):
    dict_format = {column_names[i]: arrays[i] for i in range(len(column_names))}

    df = pd.DataFrame(dict_format, index=index, dtype=np.float64)
    return df
