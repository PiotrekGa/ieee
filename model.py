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

from codes.utils import import_data, drop_columns, cross_val_score_auc, reduce_mem_usage
from codes.fe_browser import latest
from codes.fe_emails import proton, mappings, labeling
from codes.fe_cards import stats
from codes.fe_date import dates
from codes.fe_relatives import divisions
from codes.fe_categorical import pairs, wtf
from codes.prepro import prepro
from codes.fe_users import users_stats

from sklearn.feature_selection import RFECV

# %%
DATA_PATH = '../input/'
SEARCH_PARAMS = True
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

X_train = reduce_mem_usage(X_train)
X_test = reduce_mem_usage(X_test)

# %%

# %%
# columns = list(set(
#     ['dist1', 'dist2'] \
# + ['C{}'.format(i) for i in range(1,15)] \
# + ['D{}'.format(i) for i in range(1,16)] \
# + ['V' + str(i) for i in range(1,340)] \
# + ['id_' + str(i).zfill(2) for i in range(1, 12)]))

# for col in columns:
#     if col in X_train.columns:
#         X_train[col + '_' + 'trx'] = X_train.TransactionAmt * X_train[col]
#         X_test[col + '_' + 'trx'] = X_test.TransactionAmt * X_test[col]

# %% [markdown]
# ### Model and training

# %%
X_train[X_train == np.inf] = -1
X_train[X_train == -np.inf] = -1
X_test[X_test == np.inf] = -1
X_test[X_test == -np.inf] = -1

# %%
# best_params = {'num_leaves': 302,
#                  'max_depth': 157,
#                  'subsample_for_bin': 290858,
#                  'min_child_samples': 79,
#                  'reg_alpha': 0.9919573524807885,
#                  'colsample_bytree': 0.5653288564015742,
#                  'learning_rate': 0.028565794309535042}
# mod = LGBMClassifier(metric='auc',
#                      boosting_type='gbdt')
# mod.set_params(**best_params)
# rfe = RFECV(mod, step=25, min_features_to_select=150, cv=4, scoring='roc_auc', verbose=1)
# rfe.fit(X_train, y_train)

# X_train = rfe.transform(X_train)
# X_test = rfe.transform(X_test)

# %%
X_train.drop(['TransactionDT', 'TransactionAmt'], axis=1, inplace=True)
X_test.drop(['TransactionDT', 'TransactionAmt'],, axis=1, inplace=True)

# %%
model = LGBMClassifier(metric='auc',
                       n_estimators=1000,
                       boosting_type='gbdt')

# %%
SEARCH_PARAMS = False

# %%
prun = PrunedCV(N_FOLD, 0.02, minimize=False)


# %%
def objective(trial):
    
    joblib.dump(study, 'study.pkl') 
    
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 10, 1500), 
        'max_depth': trial.suggest_int('max_depth', 10, 1500), 
        'subsample_for_bin': trial.suggest_int('subsample_for_bin', 10, 3000000), 
        'min_child_samples': trial.suggest_int('min_child_samples', 2, 100000), 
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.00000000001, 10.0),
        'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.0001, 1.0),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.000001, 10.0)  
    }
    
#     params = {
#         'num_leaves': trial.suggest_int('num_leaves', 300, 310), 
#         'max_depth': trial.suggest_int('max_depth', 150, 160), 
#         'subsample_for_bin': trial.suggest_int('subsample_for_bin', 290000, 291000), 
#         'min_child_samples': trial.suggest_int('min_child_samples', 75, 82), 
#         'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.990, 0.993),
#         'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.55, 0.58),
#         'learning_rate': trial.suggest_loguniform('learning_rate', 0.02, 0.03)  
#     }
    
    
    model.set_params(**params)

    return prun.cross_val_score(model, 
                                X_train, 
                                y_train, 
                                metric='auc', 
                                shuffle=True, 
                                random_state=42)

# %%
if SEARCH_PARAMS:
    if os.path.isfile('study.pkl'):
        study = joblib.load('study.pkl')
    else:
        study = optuna.create_study()

    study.optimize(objective, timeout=60*60*6)
    joblib.dump(study, 'study.pkl')
    best_params = study.best_params
    
else:
    
    best_params = {'num_leaves': 302,
                 'max_depth': 157,
                 'subsample_for_bin': 290858,
                 'min_child_samples': 79,
                 'reg_alpha': 0.9919573524807885,
                 'colsample_bytree': 0.5653288564015742,
                 'learning_rate': 0.028565794309535042}

# %%
model.set_params(**best_params)

cross_val_score_auc(model,
                    X_train,
                    y_train,
                    n_fold=N_FOLD,
                    stratify=True,
                    shuffle=True,
                    random_state=42,
                    predict=True,
                    X_test=X_test,
                    submission=sample_submission)

# %%
# ROC accuracy: 0.9752166854560683, Train: 0.9999880028138726
# ROC accuracy: 0.978549489713329, Train: 0.9999846402519508
# ROC accuracy: 0.9775330875670358, Train: 0.9999857411401932
# ROC accuracy: 0.9779064734264544, Train: 0.9999820012259492
# ROC accuracy: 0.9759618973923397, Train: 0.9999879506811296
# ROC accuracy: 0.9760439850075724, Train: 0.999982252164232
# ROC accuracy: 0.9777317260455965, Train: 0.9999817054370517
# ROC accuracy: 0.9770691545023485, Train: 0.9999847023569826


# 0.9770015623888431

# %%
