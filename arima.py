# %%
from abc import ABC

import pandas as pd
import sklearn.ensemble as ensemble
import xgboost as xgb
import itertools as itl
import numpy as np
from statsmodels.tsa.api import ARMA
import sys


class MyARIMA(ARMA, ABC):

    def __init__(self, endog, p, q):
        super().__init__(endog, order=(p, q))
        self._my_order = (p, q)
        self._my_train = endog

    def fit(self, *args, **kwargs):
        return MyARMAResults(super().fit(disp=0),
                             train=self._my_train,
                             order=self._my_order)


class MyARMAResults:

    def __init__(self, result_obj, train, order):
        self._train = train
        self._order = order
        self._obj = result_obj

    def predict(self, *args, **kwargs):
        y_hat = []
        expanding_train = self._train.copy()
        for day_idx in range(args[0].shape[0]):
            expanding_train = expanding_train
            y_hat.append(ARMA(expanding_train, order=self._order).fit(disp=0).forecast(steps=7)[0])
        return np.vstack(y_hat).T


# %% read data

dataset = pd.read_csv("data/ml_model.csv",
                      parse_dates=["Date"],
                      index_col="Date")

ts_dataset = pd.read_csv("data/ts_model.csv",
                         parse_dates=["Date"],
                         index_col="Date")

data_y = dataset.iloc[:, -7:]
data_x = dataset.iloc[:, :-7]

train_stop_date = pd.to_datetime("2015-1-1")
valid_stop_date = pd.to_datetime("2015-1-5")

ts_train = ts_dataset.loc[ts_dataset.index <= train_stop_date]
ts_valid = ts_dataset.loc[(ts_dataset.index > train_stop_date) & (ts_dataset.index <= valid_stop_date)]
ts_test = ts_dataset.loc[ts_dataset.index >= valid_stop_date]

ts_true = pd.DataFrame(data=[ts_dataset.iloc[i:i + 7, 0].to_list() for i in range(len(ts_dataset) - 7)],
                       index=ts_dataset.index[:-7])
ts_true_train = ts_true.loc[ts_true.index <= train_stop_date]
ts_true_valid = ts_true.loc[(ts_true.index > train_stop_date) & (ts_true.index <= valid_stop_date)]
ts_true_test = ts_true.loc[ts_true.index >= valid_stop_date]

train_x: pd.DataFrame = ts_train
train_y: pd.DataFrame = ts_true_train

valid_x = ts_valid
valid_y = ts_true_valid

test_x = ts_test
test_y = ts_true_test

# %% train

model_lst = {
    # "ensemble.RandomForestRegressor": {
    #     "n_estimators": [10, 20, 30],
    #     "max_depth": [5, 10]
    # },
    "MyARIMA": {
        "endog": [ts_dataset.to_numpy(), ],
        "p": [1, 2],
        "q": [1, ]
    }
    # "xgb.XGBRegressor": {
    #     "n_estimators": [50, 100],
    #     "num_parallel_tree": [50, 100],
    #     "learning_rate": [0.1, ],
    #     "tree_method": ['hist', ],
    #     "max_depth": [5, ],
    # }
}


def param_combo_maker(param_dict):
    param_lists = [param_dict[key] for key in param_dict]
    for combo in itl.product(*param_lists):
        yield dict(zip(list(param_dict.keys()), combo))


print("Training starts")
trained_models = {}
##############################################
param_combos = {}  # list of all param combos
##############################################
Y = train_y.to_numpy()
X = train_x.to_numpy()
for model in model_lst:
    trained_models[model] = []
    param_combos[model] = list(param_combo_maker(model_lst[model]))
    for param_combo in param_combos[model]:
        model_class = eval(model)
        model_inst = model_class(**param_combo).fit(X, Y)
        trained_models[model].append(model_inst)

# %% validate
print("Validation starts")
########################################
model_errors = {}  # prediction errors
########################################
X = valid_x.to_numpy()
Y = valid_y.to_numpy()
for model in trained_models:
    model_errors[model] = []
    for combo_models in trained_models[model]:
        pred_Y = combo_models.predict(X)
        day_error = Y.T - pred_Y
        model_errors[model].append(np.linalg.norm(day_error,
                                                  ord=2,
                                                  axis=0))


def error_rmse(error_dict):
    rmse_dict = {}
    for model_ in error_dict:
        rmse_dict[model_] = {}
        for combo_ in range(len(error_dict[model_])):
            rmse_dict[model_][combo_] = error_dict[model_][combo_].mean()
    return rmse_dict


def error_std(error_dict):
    std_dict = {}
    for model_ in error_dict:
        std_dict[model_] = {}
        for combo_ in range(len(error_dict[model_])):
            std_dict[model_][combo_] = error_dict[model_][combo_].std()
    return std_dict


#################################################################
# mix and match errors key: (combo_idx, ..., combo_idx) (7) value: error scalar
# m_n_m_errors = mix_n_match(model_errors)
#################################################################
mnm_rmse = error_rmse(model_errors)
mnm_std = error_std(model_errors)


def find_best(rmse):
    model_scores = {}
    best_models = {}
    for model_ in rmse:
        model_scores[model_] = []
        best_models[model_] = None
        for mnm in rmse[model_]:
            model_scores[model_].append(rmse[model_][mnm])
        model_scores[model_] = np.array(model_scores[model_])
        best_models[model_] = list(rmse[model_].keys())[int(np.argmax(-model_scores[model_]))]
    return best_models


def mnm_2_param_combos(**kwargs):
    param_idx_lst = {}
    for model_ in kwargs:
        param_idx_lst[model_] = []
        for idx in kwargs[model_]:
            param_idx_lst[model_].append(idx)
    return param_idx_lst


best_params = find_best(mnm_rmse)
for model in best_params:
    print(model)
    print(param_combos[model][best_params[model]])

# %% test
print("Testing starts")
test_errors = {}
Y = test_y.to_numpy()
X = test_x.to_numpy()
for model in best_params:
    pred_Y = trained_models[model][best_params[model]].predict(X)
    test_errors[model] = Y.T - pred_Y
    test_errors[model] = np.linalg.norm(test_errors[model],
                                        ord=2,
                                        axis=0)
    print(f"{model} test rmse: {test_errors[model].mean()}")
    print(f"{model} test std: {test_errors[model].std()}")
