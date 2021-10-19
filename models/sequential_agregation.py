import numpy as np
import pandas as pd

from sequentior.sequential_model import SequentialModel


class BOA(SequentialModel):
    def __init__(
        self,
        frequence_training="H",
        submodel_params=None,
        horizon_prediction=2,
        df_prediction=None,
        dtype=np.float32,
    ):
        if submodel_params is None:
            submodel_params = {
                "lams": 3570,
                "n_splines": 10,
                "loss_type": "square",
                "loss_gradient": "square",
            }

        super(BOA, self).__init__(
            model_class="BOA",
            submodel_params=submodel_params,
            horizon_prediction=horizon_prediction,
            df_prediction=df_prediction,
            frequence_training=frequence_training,
        )

        self.dtype = dtype
        self.w_logits = None
        self.v = None
        self.b = None
        self.R_reg = None

    def train(self, df_train_test):
        experts, awake, y = self.get_expert(df_train_test)
        y = np.array(y)
        experts = np.array(experts)

        n = len(experts)

        if self.R_reg is None:
            self.R_reg = np.zeros(n, dtype=self.dtype)
            self.w_logits = np.zeros(n, dtype=self.dtype)
            self.v = np.zeros(n, dtype=self.dtype)
            self.b = np.zeros(n, dtype=self.dtype)

        temp, _ = self.predict(df_train_test)

        pred = np.array(temp["y"].values[-1])

        lpred = loss_function(
            pred,
            y,
            pred,
            loss_type=self.submodel_params["loss_type"],
            loss_gradient=self.submodel_params["loss_gradient"],
        )
        lexp = loss_function(
            experts,
            y,
            pred,
            loss_type=self.submodel_params["loss_type"],
            loss_gradient=self.submodel_params["loss_gradient"],
        )

        r = awake * (lpred - lexp)

        self.b = np.maximum(self.b, np.abs(r))

        self.v = self.v + r ** 2
        eta = np.minimum(1 / (2 * self.b), (np.log(n) / self.v) ** (0.5))
        eta = np.nan_to_num(eta, nan=np.nan)
        r_reg = r - eta * r ** 2
        self.R_reg = self.R_reg + r_reg
        self.w_logits = eta * self.R_reg

    def predict(self, df_train_test):
        experts, awake, _ = self.get_expert(df_train_test)

        if self.w_logits is None:
            self.w_logits = np.zeros(len(experts), dtype=self.dtype)
        w_logits = self.w_logits + np.log(awake)
        w_logits = w_logits - np.max(w_logits)
        w = np.exp(w_logits)
        w /= np.sum(w)
        pred = np.dot(experts, w)

        dates = df_train_test.index.values[-1]

        y_pred = pd.DataFrame(pred, index=[dates], columns=["y"])
        return y_pred, w

    def get_expert(self, df_train_test):

        experts = df_train_test.iloc[-1].drop("y").values
        experts = experts.astype(self.dtype)
        awake = (~np.isnan(experts)).astype(self.dtype)
        y = float(df_train_test["y"].iloc[-1])
        experts = np.nan_to_num(experts)
        return experts, awake, y


class MLpol(SequentialModel):
    def __init__(
        self,
        submodel_params=None,
        horizon_prediction=2,
        frequence_training="H",
        df_prediction=None,
        dtype=np.float32,
    ):
        if submodel_params is None:
            submodel_params = {
                "lams": 3570,
                "n_splines": 10,
                "loss_type": "square",
                "loss_gradient": "square",
            }
        super().__init__(
            model_class="MLpol",
            submodel_params=submodel_params,
            horizon_prediction=horizon_prediction,
            df_prediction=df_prediction,
            frequence_training=frequence_training,
        )
        self.dtype = dtype
        self.w_logits = None
        self.v = None
        self.b = None
        self.R = None
        self.eta = None

    def train(self, df_train_test):
        experts, awake, y = self.get_expert(df_train_test)
        n = len(experts)
        if self.R is None:
            self.R = np.zeros(n, dtype=self.dtype)
            self.w_logits = np.zeros(n, dtype=self.dtype)
            self.v = np.zeros(n, dtype=self.dtype)
            self.b = np.zeros(n, dtype=self.dtype)
            self.eta = np.zeros(n) - np.inf

        pred = self.predict(df_train_test)["y"].values[-1]
        lpred = loss_function(
            pred,
            y,
            pred,
            loss_type=self.submodel_params["loss_type"],
            loss_gradient=self.submodel_params["loss_gradient"],
        )
        lexp = loss_function(
            experts,
            y,
            pred,
            loss_type=self.submodel_params["loss_type"],
            loss_gradient=self.submodel_params["loss_gradient"],
        )
        r = awake * (lpred - lexp)
        self.R += r
        r_square = r ** 2

        new_b = np.maximum(self.b, r_square)

        self.v = self.v + r_square
        self.eta = 1 / (1 / self.eta + r_square + new_b - self.b)
        self.eta = np.nan_to_num(self.eta, nan=np.nan)
        self.b = new_b

        self.w_logits = self.eta * np.maximum(self.R, 0)

    def predict(self, df_train_test):
        experts, awake, _ = self.get_expert(df_train_test)
        n = len(experts)
        if self.w_logits is None:
            self.w_logits = np.zeros(len(experts), dtype=self.dtype)
        denom = np.sum(awake * self.w_logits)
        if denom == 0:
            p = np.ones(n) / n
        else:
            p = awake * self.w_logits / denom
        pred = np.dot(experts, p)

        dates = df_train_test.index.values[-1]

        y_pred = pd.DataFrame(pred, index=[dates], columns=["y"])
        return y_pred

    def get_expert(self, df_train_test):

        experts = df_train_test.iloc[-1].drop("y").values
        experts = experts.astype(self.dtype)
        awake = (~np.isnan(experts)).astype(self.dtype)
        y = float(df_train_test["y"].iloc[-1])
        experts = np.nan_to_num(experts)
        return experts, awake, y


def loss_function(x, y, pred=None, loss_type="square", loss_gradient=False):
    if not loss_gradient:
        if loss_type == "square":

            return np.square(x - y)
        elif loss_type == "absolute":
            return np.abs(x - y)
        elif loss_type == "percentage":
            return np.abs(x - y) / np.abs(y)
        else:
            raise NotImplementedError(f"lost type {loss_type} is not understood")
    else:
        if loss_type == "square":
            return 2 * (pred - y) * x
        elif loss_type == "absolute":
            return np.sign(pred - y) * x
        elif loss_type == "percentage":
            return np.sign(pred - y) * x / np.abs(y)
        else:
            raise NotImplementedError(f"lost type {loss_type} is not understood")
