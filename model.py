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
from datetime import datetime

import lightgbm as lgb
import optuna

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

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
SEARCH_PARAMS = True


# %% {"_cell_guid": "79c7e3d0-c299-4dcb-8224-4455121ee9b0", "_uuid": "d629ff2d2480ee46fbb7e2d37f6b5fab8052498a"}
train, test, sample_submission = utils.import_data(DATA_PATH)

### Some Feature Engineering
train, test = fe_users.users_stats(train, test)

train, test = utils.drop_columns(train, test)

train, test = fe_browser.latest(train, test)

train, test = fe_emails.proton(train, test)

train['nulls1'] = train.isna().sum(axis=1)
test['nulls1'] = test.isna().sum(axis=1)

train, test = fe_emails.mappings(train, test)
train, test = fe_emails.labeling(train, test)

train, test = fe_cards.stats(train, test)

train, test = fe_relatives.divisions(train, test)

train, test = fe_date.dates(train, test)

train, test = fe_categorical.pairs(train, test)
train, test = fe_categorical.wtf(train, test)


# %%
train.shape

# %%
X_train = train.copy()
X_test = test.copy()

del train, test

#fill in mean for floats
X_train, X_test = prepro.prepro(X_train, X_test)

y_train = X_train['isFraud'].copy()
X_train = X_train.drop('isFraud', axis=1)

# %%
X_train = utils.reduce_mem_usage(X_train)

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
    
    print('')
    return np.mean(model_scores)


# %%
def objective(trial):
    
    joblib.dump(study, 'study.pkl')
    
    X_train_, X_test_, y_train_, y_test_ = train_test_split(X_train, y_train, stratify=y_train)
    
    dtrain = lgb.Dataset(X_train_, label=y_train_)
    dtest = lgb.Dataset(X_test_, label=y_test_)

    param = {
        'objective': 'binary',
        'metric': 'binary_error',
        'verbosity': -1,
        'num_leaves': trial.suggest_int('num_leaves', 10, 500),
        'max_depth': trial.suggest_int('max_depth', 10, 300), 
        'n_estimators': trial.suggest_int('n_estimators', 50, 2000), 
        'subsample_for_bin': trial.suggest_int('subsample_for_bin', 1000, 500000), 
        'min_child_samples': trial.suggest_int('min_child_samples', 2, 200), 
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.00001, 10.0),
        'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.0001, 1.0),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.000001, 10.0)   
    }

    # Add a callback for pruning.
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, 'binary_error')
    gbm = lgb.train(
        param, dtrain, valid_sets=[dtest], verbose_eval=False, callbacks=[pruning_callback])

    preds = gbm.predict(X_test_)
    
    del X_train_, X_test_, y_train_
    
    gc.collect()
    
    return - roc_auc_score(y_test_, preds)


# %%
if SEARCH_PARAMS:
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    study.optimize(objective, timeout=60*60*6)

    joblib.dump(study, 'study.pkl')

    trials_df = pd.DataFrame([trial.value, trial.params] for trial in study.trials)
    trials_df.columns = ['value', 'params']
    trials_df.sort_values(by='value', inplace=True)
    params_for_cv = 10
    trials_df = trials_df.iloc[:params_for_cv,:]

    model = lgb.LGBMClassifier(metric='auc')
    n_folds = 8

    best_params = None
    best_value = 0

    for params in trials_df.params:
        model.set_params(**params)
        score = cross_val_score2(model, X_train, y_train, n_folds)

        if score > best_value:
            best_value = score
            best_params = params

        gc.collect()
        
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
n_fold = 8
folds = StratifiedKFold(n_splits=n_fold, shuffle=True)

model = lgb.LGBMClassifier(metric='auc')
model.set_params(**best_params)

model_scores = []
model_scores_tr = []
for train_index, valid_index in folds.split(X_train, y_train):
    
    X_train_, X_valid = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_train_, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]
    model.fit(X_train_,y_train_)
    train_val = model.predict_proba(X_train_)[:,1]
    model_scores_tr.append(roc_auc_score(y_train_, train_val))
    del X_train_,y_train_, train_val
    pred = model.predict_proba(X_test)[:,1]
    val = model.predict_proba(X_valid)[:,1]
    
    del X_valid
    model_scores.append(roc_auc_score(y_valid, val))
    print('ROC accuracy: {}, ROC train: {}'.format(model_scores[-1], model_scores_tr[-1]))
    del val, y_valid
    submission['isFraud'] = submission['isFraud'] + pred / n_fold
    del pred
    gc.collect()
    
model_score = np.round(np.mean(model_scores),4)
print(model_score)
print(params)

# %%
timestamp = str(int(datetime.timestamp(datetime.now())))
submission.to_csv('{}_submission_{}.csv'.format(timestamp, str(model_score)))
