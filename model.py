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
# # !rm -r ieee
# # !rm -r codes
# # !git clone https://github.com/PiotrekGa/ieee.git
# # !mv ieee/codes .

# %%
import os
import gc
import numpy as np
import pandas as pd
import joblib

from lightgbm import LGBMClassifier
import optuna

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from codes import utils
from codes import fe_browser
from codes import fe_emails
from codes import fe_cards
from codes import fe_date
from codes import fe_relatives
from codes import fe_categorical
from codes import prepro
from codes import fe_users

# %%
DATA_PATH = '../input/'
SEARCH_PARAMS = False


# %% {"_cell_guid": "79c7e3d0-c299-4dcb-8224-4455121ee9b0", "_uuid": "d629ff2d2480ee46fbb7e2d37f6b5fab8052498a"}
train, test, sample_submission = utils.import_data(DATA_PATH)


# %% [markdown]
# ### Some Feature Engineering
#
# drop columns, count encoding, aggregation, fillna

# %%
train, test = fe_users.users_stats(train, test)

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
submission = sample_submission.copy()
submission['isFraud'] = 0

# %%
model = LGBMClassifier(metric='auc')


# %%
def cross_val_score2(model, X_train, y_train, n_fold):
    
    folds = StratifiedKFold(n_splits=n_fold, shuffle=True)
    
    model_scores = []
    for train_index, valid_index in folds.split(X_train, y_train):
        X_train_, X_valid = X_train.iloc[train_index], X_train.iloc[valid_index]
        y_train_, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]
        model.fit(X_train_,y_train_)
        del X_train_,y_train_
        val = model.predict_proba(X_valid)[:,1]
        del X_valid
        model_scores.append(roc_auc_score(y_valid, val))
        print('ROC accuracy: {}'.format(model_scores[-1]))
        del val, y_valid
    
    return np.mean(model_scores)


# %%
def objective(trial):
    
    joblib.dump(study, 'study.pkl')
    
    num_leaves = trial.suggest_int('num_leaves', 110, 120) 
    max_depth = trial.suggest_int('max_depth', 70, 80) 
    n_estimators = trial.suggest_int('n_estimators', 270, 280) 
    subsample_for_bin = trial.suggest_int('subsample_for_bin', 87000, 88044) 
    min_child_samples = trial.suggest_int('min_child_samples', 1000, 1101) 
    reg_alpha = trial.suggest_uniform('reg_alpha', 0.902, 0.953) 
    colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.6567, 0.75678) 
    learning_rate = trial.suggest_loguniform('learning_rate', 0.04, 0.05)   

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

    return - cross_val_score2(model, X_train, y_train, 8)


# %%
study = optuna.create_study()
study.optimize(objective, n_trials=5)


# %%
def objective(trial):
    
    joblib.dump(study, 'study.pkl')
    
    num_leaves = trial.suggest_int('num_leaves', 2, 500) 
    max_depth = trial.suggest_int('max_depth', 2, 300) 
    n_estimators = trial.suggest_int('n_estimators', 100, 2000) 
    subsample_for_bin = trial.suggest_int('subsample_for_bin', 100_000, 500_000) 
    min_child_samples = trial.suggest_int('min_child_samples', 20, 10000) 
    reg_alpha = trial.suggest_uniform('reg_alpha', 0.0, 2.0) 
    colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.5, 1.0) 
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

    return - cross_val_score2(model, X_train, y_train, 8)


# %%
if SEARCH_PARAMS:

    if os.path.isfile('study.pkl'):
        study = joblib.load('study.pkl')
    else:
        study = optuna.create_study()
    study.optimize(objective, timeout=60*60*9)
    
    params = study.best_params

else:
    
    params = {'num_leaves': 302,
             'max_depth': 157,
             'n_estimators': 966,
             'subsample_for_bin': 290858,
             'min_child_samples': 79,
             'reg_alpha': 0.9919573524807885,
             'colsample_bytree': 0.5653288564015742,
             'learning_rate': 0.028565794309535042}

# %%
model = LGBMClassifier(metric='auc')
model.set_params(**params)

n_fold = 8
folds = StratifiedKFold(n_splits=n_fold, shuffle=True)

model_scores = []
for train_index, valid_index in folds.split(X_train, y_train):
    X_train_, X_valid = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_train_, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]
    model.fit(X_train_,y_train_)
    del X_train_,y_train_
    pred=model.predict_proba(X_test)[:,1]
    val=model.predict_proba(X_valid)[:,1]
    del X_valid
    model_scores.append(roc_auc_score(y_valid, val))
    print('ROC accuracy: {}'.format(model_scores[-1]))
    del val, y_valid
    submission['isFraud'] = submission['isFraud'] + pred / n_fold
    del pred
    
model_score = np.round(np.mean(model_scores),4)
print(model_score)
print(params)

# %%
submission.to_csv('submission_{}.csv'.format(str(model_score)))
