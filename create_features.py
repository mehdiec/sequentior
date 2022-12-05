from datetime import timedelta
import logging

import numpy as np
import pandas as pd

from dataframe_formatting import (
    create_features,
    dataframe_model,
    turn_array_into_dataframe,
)
from models.sequential_model import SequentialModel


class FeatureCreation(SequentialModel):
    def __init__(
        self,
        frequence_training=None,
        submodel_params=None,
        df_prediction=None,
        horizon_prediction=None,
    ):

        super(FeatureCreation, self).__init__(
            model_class=None,
            frequence_training=frequence_training,
            df_prediction=df_prediction,
        )
        if submodel_params is None:
            submodel_params = {
                "span_evm": 240,
                "hour_shift_list": [24, 167, 191],
                "horizon_shift": True,
                "standardize": False,
                "harmonic": 0,
            }
        self.horizon_prediction = horizon_prediction

        self.submodel_params = submodel_params

    def train(self, df_train):
        pass

    def predict(self, df_train):

        X_int = create_features(
            df_train, harmonic=self.submodel_params["harmonic"], label=["y"]
        )

        time_stamp = df_train.index.values
        df_train = df_train.reset_index()
        tmp_1 = np.maximum(X_int["TMP"] - 17, 0)
        tmp_2 = np.maximum(17 - X_int["TMP"], 0)
        evm_temperature = X_int["TMP"].ewm(span=self.submodel_params["span_evm"]).mean()

        y = X_int["y"]

        hour_shift_dict = {}
        for i, hour_shift in enumerate(self.submodel_params["hour_shift_list"]):
            if hour_shift < self.horizon_prediction:
                logging.warning(
                    "Horizon exceeds shift, shift will be replace by horizon "
                )
                hour_shift = self.horizon_prediction
            hour_shift_dict[f"conso_shift_{i}"] = y.shift(hour_shift)
        df_hour_shift = pd.DataFrame(hour_shift_dict, index=time_stamp)
        X_fourier = np.array([evm_temperature, tmp_1, tmp_2])
        column_names = ["evm_temperature", "tmp_1", "tmp_2"]
        if self.submodel_params["horizon_shift"]:
            horizon_shift = y.shift(self.horizon_prediction)
            X_fourier = np.array([horizon_shift, evm_temperature])
            column_names = [
                "horizon_shift",
                "evm_temperature",
            ]

        arrays = X_fourier
        df_fourier = turn_array_into_dataframe(arrays, column_names, time_stamp)
        df_features = pd.concat([X_int, df_fourier, df_hour_shift], axis=1, sort=False)
        if self.submodel_params["standardize"]:
            df_bias = pd.DataFrame([1 for _ in range(len(y))], index=df_features.index)

            df_normalize = 2 * df_features / df_features.max()
            df_features = pd.concat(
                [df_normalize, -df_normalize.drop("y", axis=1), df_bias], axis=1
            )

        return df_features

    def fit_predict(self, df_train):
        return self.predict(df_train)


if __name__ == "__main__":
    df = dataframe_model(
        r"/home/dehk/sequentior/data/conso/eCO2mix_RTE_Ile-de-France_Annuel-Definitif",
        "/home/dehk/sequentior/data/weather/0715609999",
    )
    ft_creation = FeatureCreation(horizon_prediction=20)  # timedelta(hours=20))
    df_ft = ft_creation.predict(df)
    print(df_ft)
