from datetime import timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd

# from dataframe_formatting import dataframe_model, make_train_test_all


def get_model_class(model_class):
    from linear_models import (
        SequentialGAMModel,
        SequentialLinearModel,
    )
    from sequential_agregation import BOA, MLpol

    if model_class == "sequential_gam":
        return SequentialGAMModel
    if model_class == "sequential_linear":
        return SequentialLinearModel
    if model_class == "BOA":
        return BOA
    if model_class == "MLpol":
        return MLpol


class SequentialModel:
    def __init__(
        self,
        model_class,
        frequence_training=None,
        submodel_params=None,
        df_prediction=None,
        horizon_prediction=None,
        weights=None,
    ):
        """[summary]

        Args:
            frequence_training ([type], optional): [description]. Defaults to None.
            submodel_params ([type], optional): [description]. Defaults to None.
            df_prediction ([type], optional): [description]. Defaults to None.
            model ([type], optional): [description]. Defaults to None.
            horizon_prediction ([type], optional): [description]. Defaults to None.
            span_evm (int, optional): [description]. Defaults to 240.
            hour_shift (int, optional): [description]. Defaults to 24.
        """
        self.model_class = model_class
        if submodel_params is None:
            submodel_params = {}
        self.submodel_params = dict(submodel_params)
        if df_prediction is None:
            df_prediction = pd.DataFrame(columns=["y"])

        self.df_prediction = df_prediction
        if frequence_training is None:
            frequence_training = pd.DateOffset(days=7, hour=19)
        if horizon_prediction is None:
            horizon_prediction = pd.DateOffset(days=1)
        if weights is None:
            weights = None
        self.weights = weights

        self.frequence_training = frequence_training
        self.horizon_prediction = horizon_prediction

    def train(self, df_train_test):
        raise NotImplementedError

    def predict(self, df_train_test):
        raise NotImplementedError

    def fit_predict(self, df_train_test):
        if self.weights is None:
            self.weights = np.array([0 for _ in range(len(df_train_test.columns) - 1)])

        datelist = pd.date_range(
            df_train_test.index[0],
            df_train_test.index[-1],
            freq=self.frequence_training,
        ).tolist()
        if not datelist:

            datelist = pd.date_range(
                df_train_test.index[-1],
                df_train_test.index[0],
                freq=self.frequence_training,
            ).tolist()

        if df_train_test.index[-1] != datelist[-1]:
            datelist.append(df_train_test.index[-1])

        for i in range(len(datelist[:]) - 1):

            end_train_date = datelist[i] - pd.to_timedelta(self.horizon_prediction)
            start_pred_date = datelist[i]
            end_pred_date = datelist[i + 1]
            if i == 0:
                df_train = df_train_test.loc[:end_train_date]
            else:
                df_train = df_train_test.loc[start_train_date:end_train_date]

            if len(df_train) > 0:
                if len(df_train.dropna()):
                    self.train(df_train)

            df_test = df_train_test.loc[start_pred_date:end_pred_date]
            if len(df_test) > 0:
                if self.model_class == "BOA":
                    predtiction, w = self.predict(df_test)
                    self.df_prediction = self.df_prediction.append(predtiction)
                    self.weights = np.vstack((self.weights, w))

                else:

                    self.df_prediction = self.df_prediction.append(
                        self.predict(df_test)
                    )
            start_train_date = end_train_date

    def save_model(self, model_path):
        model_path = Path(model_path)
        model_path.mkdir(parents=True, exist_ok=True)
        self.df_prediction.to_pickle(model_path / "df_prediction.pkl")

        params = {
            "submodel_params": self.submodel_params,
            "horizon_prediction": self.horizon_prediction,
            "frequence_training": None,
            "model_class": self.model_class,
        }
        with open(model_path / "params.json", "w") as fp:
            json.dump(params, fp)

    @staticmethod
    def load_model(model_path):
        model_path = Path(model_path)
        with open(model_path / "params.json", "r") as fp:
            params = json.load(fp)
        df_prediction = pd.read_pickle(model_path / "df_prediction.pkl")
        model_class = get_model_class(params["model_class"])
        print(model_class)
        return model_class(
            frequence_training=params["frequence_training"],
            submodel_params=params["submodel_params"],
            df_prediction=df_prediction,
            horizon_prediction=params["horizon_prediction"],
        )


if __name__ == "__main__":
    # df = dataframe_model(
    #     r"/home/dehk/sequentior/data/conso/eCO2mix_RTE_Ile-de-France_Annuel-Definitif",
    #     "/home/dehk/sequentior/data/weather/0715609999",
    # )

    # X, y, time_list = make_train_test_all(df)
    model = SequentialModel("BOA", horizon_prediction=timedelta(hours=19))
    # model.fit_predict(df)
