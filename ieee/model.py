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

from lightgbm import LGBMClassifier
import optuna

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_score

from ieee import utils
from ieee import fe_browser
from ieee import fe_emails
from ieee import fe_cards
from ieee import fe_date
from ieee import fe_relatives
from ieee import fe_categorical
from ieee import prepro

# %%
DATA_PATH = '../input/'


# %% {"_cell_guid": "79c7e3d0-c299-4dcb-8224-4455121ee9b0", "_uuid": "d629ff2d2480ee46fbb7e2d37f6b5fab8052498a"}
train, test, sample_submission = utils.import_data(DATA_PATH)


# %% [markdown]
# ### Some Feature Engineering
#
# drop columns, count encoding, aggregation, fillna

# %%
train, test = utils.drop_columns(train, test)

# %%
train, test = fe_browser.latest(train, test)

# %%
train, test = fe_emails.proton(train, test)

train['nulls1'] = train.isna().sum(axis=1)
test['nulls1'] = test.isna().sum(axis=1)

train, test = fe_emails.mappings(train, test)
train, test = fe_emails.labeling(train, test)

# %%
train, test = fe_cards.stats(train, test)

# %%
train, test = fe_relatives.divisions(train, test)

# %%
train, test = fe_date.dates(train, test)


# %%
train, test = fe_categorical.pairs(train, test)
train, test = fe_categorical.wtf(train, test)


# %%
y_train = train['isFraud'].copy()


X_train = train.drop('isFraud', axis=1)
X_test = test.copy()

del train, test

#fill in mean for floats
X_train, X_test = prepro.prepro(X_train, X_test)

# %% [markdown]
# ### Model and training

# %%
submission=sample_submission.copy()
submission['isFraud'] = 0

# %%
model = LGBMClassifier(metric='auc')


# %%
def objective(trial):
    
    joblib.dump(study, 'study.pkl')
    
    num_leaves = trial.suggest_int('num_leaves', 2, 200) 
    max_depth = trial.suggest_int('max_depth', 2, 100) 
    n_estimators = trial.suggest_int('n_estimators', 10, 500) 
    subsample_for_bin = trial.suggest_int('subsample_for_bin', 2000, 300_000) 
    min_child_samples = trial.suggest_int('min_child_samples', 20, 1000) 
    reg_alpha = trial.suggest_uniform('reg_alpha', 0.0, 1.0) 
    colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.6, 1.0) 
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-0)   

    params = {
        'num_leaves': num_leaves,
        'max_depth': max_depth,
        'n_estimators': n_estimators,
        'subsample_for_bin': subsample_for_bin,
        'min_child_samples': min_child_samples,
        'reg_alpha': reg_alpha,
        'colsample_bytree': colsample_bytree,
        'learning_rate': learning_rate
    }
    
    model.set_params(**params)

    return - np.mean(cross_val_score(model, X_train, y_train, cv=4, scoring='roc_auc'))


# %%
if os.path.isfile('study.pkl'):
    study = joblib.load('study.pkl')
else:
    study = optuna.create_study()
study.optimize(objective, timeout=60*60*8)

# %%
print(study.best_params)

# %%
n_fold = 8
folds = KFold(n_splits=n_fold, shuffle=True)

for train_index, valid_index in folds.split(X_train):
    model.set_params(**study.best_params)
    X_train_, X_valid = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_train_, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]
    model.fit(X_train_,y_train_)
    del X_train_,y_train_
    pred=model.predict_proba(X_test)[:,1]
    val=model.predict_proba(X_valid)[:,1]
    del X_valid
    print('ROC accuracy: {}'.format(roc_auc_score(y_valid, val)))
    del val, y_valid
    submission['isFraud'] = submission['isFraud'] + pred / n_fold
    del pred

# %%
submission.to_csv('submission.csv')
