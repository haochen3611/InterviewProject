# %%
import pandas as pd
import sklearn.ensemble as ensemble
import xgboost as xgb
import itertools as itl
import numpy as np


# %% read data

dataset = pd.read_csv("data/ml_model.csv",
                      parse_dates=["Date"],
                      index_col="Date")

data_y = dataset.iloc[:, -7:]
data_x = dataset.iloc[:, :-7]

train_stop_date = pd.to_datetime("2015-1-1")
valid_stop_date = pd.to_datetime("2018-1-1")

train_x: pd.DataFrame = data_x.loc[data_x.index <= train_stop_date]
train_y: pd.DataFrame = data_y.loc[data_y.index <= train_stop_date]

valid_x = data_x.loc[(data_x.index > train_stop_date) & (data_x.index <= valid_stop_date)]
valid_y = data_y.loc[(data_y.index > train_stop_date) & (data_y.index <= valid_stop_date)]

test_x = data_x.loc[data_x.index >= valid_stop_date]
test_y = data_y.loc[data_y.index >= valid_stop_date]
# %% train

model_lst = {
    "ensemble.RandomForestRegressor": {
        "n_estimators": [10, 20, 30],
        "max_depth": [5, 10]
    },
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


trained_models = {}
##############################################
param_combos = {}  # list of all param combos
##############################################

for model in model_lst:
    trained_models[model] = []
    param_combos[model] = list(param_combo_maker(model_lst[model]))
    for param_combo in param_combos[model]:
        trained_models[model].append([])
        Y = train_y.to_numpy()
        X = train_x.to_numpy()
        for day_col in range(Y.shape[1]):
            model_inst: ensemble.RandomForestRegressor = eval(model)(**param_combo)
            model_inst.fit(X, Y[:, day_col])
            trained_models[model][-1].append(model_inst)

# %% validate

########################################
model_errors = {}  # prediction errors
########################################

for model in trained_models:
    model_errors[model] = []
    for combo_models in trained_models[model]:
        model_errors[model].append([])
        Y = train_y.to_numpy()
        X = train_x.to_numpy()
        for day_model in range(Y.shape[1]):
            pred_Y = combo_models[day_model].predict(X)
            day_error = Y[:, day_model] - pred_Y
            model_errors[model][-1].append(day_error)


def mix_n_match(error_dict):
    m_n_m_error_dict = {}
    for model_ in error_dict:
        combo_idx = list(range(len(error_dict[model_])))
        m_n_m_error_dict[model_] = {}
        m_n_m_s = itl.product(combo_idx, repeat=7)
        for m_n_m in m_n_m_s:
            m_n_m_error = []
            for day in range(7):
                m_n_m_error.append(error_dict[model_][m_n_m[day]][day])
            m_n_m_error_dict[model_][tuple(m_n_m)] = np.linalg.norm(np.vstack(m_n_m_error),
                                                                    ord=2,
                                                                    axis=0)
    return m_n_m_error_dict


def error_rmse(error_dict):
    rmse_dict = {}
    for model_ in error_dict:
        rmse_dict[model_] = {}
        for combo_ in error_dict[model_]:
            rmse_dict[model_][combo_] = error_dict[model_][combo_].mean()
    return rmse_dict


def error_std(error_dict):
    std_dict = {}
    for model_ in error_dict:
        std_dict[model_] = {}
        for combo_ in error_dict[model_]:
            std_dict[model_][combo_] = error_dict[model_][combo_].std()
    return std_dict

#################################################################
# mix and match errors key: (combo_idx, ..., combo_idx) (7) value: error scalar
m_n_m_errors = mix_n_match(model_errors)
#################################################################
mnm_rmse = error_rmse(m_n_m_errors)
mnm_std = error_std(m_n_m_errors)


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


best_params = mnm_2_param_combos(**find_best(mnm_rmse))
for model in best_params:
    print(model)
    for combo_idx in best_params[model]:
        print(param_combos[model][combo_idx])

# %% test

##########################################
test_errors = {}
###########################################
Y = test_y.to_numpy()
X = test_x.to_numpy()
for model in best_params:
    test_errors[model] = []
    for day, day_combo in enumerate(best_params[model]):
        pred_Y = trained_models[model][day_combo][day].predict(X)
        test_errors[model].append(Y[:, day] - pred_Y)
    test_errors[model] = np.linalg.norm(np.vstack(test_errors[model]),
                                        ord=2,
                                        axis=0)
    print(f"{model} test rmse: {test_errors[model].mean()}")
    print(f"{model} test std: {test_errors[model].std()}")


