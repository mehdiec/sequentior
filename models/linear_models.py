import numpy as np
import pandas as pd
from pygam.pygam import LinearGAM
from pygam.terms import s
from sklearn.linear_model import LinearRegression

from .sequential_model import SequentialModel


class SequentialGAMModel(SequentialModel):
    def __init__(
        self,
        frequence_training=None,
        submodel_params=None,
        df_prediction=None,
        horizon_prediction=None,
    ):
        super(SequentialGAMModel, self).__init__(
            model_class="sequential_gam",
            frequence_training=frequence_training,
            submodel_params=submodel_params,
            df_prediction=df_prediction,
            horizon_prediction=horizon_prediction,
        )
        self.model = None

    def train(self, df_train):

        df_train_features = df_train.dropna()

        X_train = df_train_features.drop("y", axis=1).values
        y_train = df_train_features["y"].values
        if type(self.submodel_params["lams"]) is not list:
            tot = None
            for i, feature in enumerate(
                df_train_features.drop("y", axis=1).columns.to_list()
            ):

                if "pos" in feature:
                    term = s(i, basis="cp")
                else:
                    term = s(i)
                if i == 0:
                    tot = term
                else:
                    tot += term

            self.model = LinearGAM(
                tot,
                n_splines=self.submodel_params["n_splines"],
                lam=self.submodel_params["lams"],
            )

            self.model.fit(X_train, y_train)
        if self.submodel_params["gridsearch"]:
            self.model = LinearGAM(
                n_splines=self.submodel_params["n_splines"],
            )

            self.model.gridsearch(X_train, y_train, lam=self.submodel_params["lams"])

    def predict(self, df_train_test):
        df_test_features = df_train_test
        dates = df_test_features.index
        X_test = df_train_test.drop("y", axis=1)
        X_test = X_test.astype(np.float32)
        not_nan = ~np.any(np.isnan(X_test), axis=1)
        y_pred = pd.DataFrame(
            [np.nan for _ in range(len(dates))], index=dates, columns=["y"]
        )

        if self.model is not None:
            y_pred.loc[not_nan, "y"] = self.model.predict(X_test.loc[not_nan])

        return y_pred


class SequentialLinearModel(SequentialModel):
    def __init__(
        self,
        frequence_training=None,
        submodel_params=None,
        df_prediction=None,
        horizon_prediction=None,
    ):
        super(SequentialLinearModel, self).__init__(
            model_class="sequential_linear",
            frequence_training=frequence_training,
            submodel_params=submodel_params,
            df_prediction=df_prediction,
            horizon_prediction=horizon_prediction,
        )
        self.model = LinearRegression(warm_start=True)

    def train(self, df_train):

        df_train_features = df_train.dropna()
        X_train = df_train_features.drop("y", axis=1)
        y_train = df_train_features["y"]
        self.model.fit(X_train.values, y_train)

    def predict(self, df_train_test):

        dates = df_train_test.index
        X_test = df_train_test.drop("y", axis=1)
        X_test = X_test.astype(np.float32)
        not_nan = ~np.any(np.isnan(X_test), axis=1)
        y_pred = pd.DataFrame(
            [np.nan for _ in range(len(dates))], index=dates, columns=["y"]
        )
        if self.model is not None:
            y_pred.loc[not_nan, "y"] = self.model.predict(X_test.loc[not_nan])

        return y_pred
