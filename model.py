# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.0.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os
import gc
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from lightgbm import LGBMClassifier
import optuna
from prunedcv import PrunedCV

from codes.utils import import_data, drop_columns, cross_val_score_auc
from codes.fe_browser import latest
from codes.fe_emails import proton, mappings, labeling
from codes.fe_cards import stats
from codes.fe_date import dates
from codes.fe_relatives import divisions
from codes.fe_categorical import pairs, wtf
from codes.prepro import prepro
from codes.fe_users import users_stats

# %%
DATA_PATH = '../input/'
SEARCH_PARAMS = False
N_FOLD = 8


# %% {"_cell_guid": "79c7e3d0-c299-4dcb-8224-4455121ee9b0", "_uuid": "d629ff2d2480ee46fbb7e2d37f6b5fab8052498a"}
train, test, sample_submission = import_data(DATA_PATH)


# %% [markdown]
# ### Some Feature Engineering
#
# drop columns, count encoding, aggregation, fillna

# %%
train, test = users_stats(train, test)

train, test = drop_columns(train, test)

train, test = latest(train, test)

train, test = proton(train, test)

train['nulls1'] = train.isna().sum(axis=1)
test['nulls1'] = test.isna().sum(axis=1)

train, test = mappings(train, test)
train, test = labeling(train, test)

train, test = stats(train, test)

train, test = divisions(train, test)

train, test = dates(train, test)

train, test = pairs(train, test)
train, test = wtf(train, test)

y_train = train['isFraud'].copy()


X_train = train.drop('isFraud', axis=1)
X_test = test.copy()

del train, test

#fill in mean for floats
X_train, X_test = prepro(X_train, X_test)

# %% [markdown]
# ### Model and training

# %%
model = LGBMClassifier(metric='auc',
                       n_estimators=2000,
                       boosting_type='gbdt',
                       is_unbalance=True,)

# %%
prun = PrunedCV(N_FOLD, 0.02, minimize=False)

def objective(trial):
    
    joblib.dump(study, 'study.pkl') 
    
    params = {
        'max_depth': trial.suggest_int('max_depth', 10, 5000), 
        'subsample_for_bin': trial.suggest_int('subsample_for_bin', 1000, 3000000), 
        'min_child_samples': trial.suggest_int('min_child_samples', 2, 100000), 
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.00000000001, 10.0),
        'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.0001, 1.0),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.000001, 10.0)  
    }
    
    model = LGBMClassifier()
    model.set_params(**params)

    return prun.cross_val_score(model, 
                                X_train, 
                                y, 
                                metric='auc', 
                                shuffle=True, 
                                random_state=42)


# %%
if SEARCH_PARAMS:
    if os.path.isfile('study.pkl'):
        study = joblib.load('study.pkl')
    else:
        study = optuna.create_study()

    study.optimize(objective, timeout=60*60*7)
    joblib.dump(study, 'study.pkl')
    best_params = study.best_params
    
else:
    
    best_params = {'num_leaves': 302,
                   'max_depth': 157,
                   'n_estimators': 1200,
                   'subsample_for_bin': 290858,
                   'min_child_samples': 79,
                   'reg_alpha': 1.0919573524807885,
                   'colsample_bytree': 0.5653288564015742,
                   'learning_rate': 0.028565794309535042}

# %%
model = LGBMClassifier(metric='auc',
                       n_estimators=2000,
                       boosting_type='gbdt',
                       is_unbalance=True,)

model.set_params(**best_params)

cross_val_score_auc(model,
                    X_train,
                    y_train,
                    n_fold=N_FOLD,
                    shuffle=False,
                    random_state=42,
                    predict=True,
                    X_test=X_test,
                    submission=sample_submission)

# %%
