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

from codes.utils import cross_val_score_auc

# %%
SEARCH_PARAMS = False
N_FOLD = 8
BOOSTING = 'gbdt'
RANDOM_STATE = 42


# %%
y_train = joblib.load('y_train.pkl')
X_train = joblib.load('features_train.pkl')
X_test = joblib.load('features_test.pkl')
sample_submission = pd.read_csv('../input/sample_submission.csv', index_col=0)

# %% [markdown]
# ### Model and training

# %%
model = LGBMClassifier(metric='auc',
                       boosting_type=BOOSTING)

# %%
prun = PrunedCV(N_FOLD, 0.005, splits_to_start_pruning=3, minimize=False)


# %%
def objective(trial):
    
    joblib.dump(study, 'study_{}.pkl'.format(BOOSTING)) 

    
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 10, 1500), 
        'max_depth': trial.suggest_int('max_depth', 10, 1000), 
        'subsample_for_bin': trial.suggest_int('subsample_for_bin', 1000, 5000000), 
        'min_child_samples': trial.suggest_int('min_child_samples', 200, 100000), 
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.00000000001, 10.0),
        'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.0001, 1.0),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.00001, 2.0),
        'n_estimators': trial.suggest_int('n_estimators', 1000, 2000)
    }
    
    
    
    print(params)
    
    model.set_params(**params)
    return prun.cross_val_score(model, 
                                X_train, 
                                y_train, 
                                metric='auc', 
                                shuffle=True, 
                                random_state=RANDOM_STATE)

# %%
if SEARCH_PARAMS:
    if os.path.isfile('study_{}.pkl'.format(BOOSTING)):
        study = joblib.load('study_{}.pkl'.format(BOOSTING))
    else:
        study = optuna.create_study()

    study.optimize(objective, timeout=60 * 60 * 15)
    joblib.dump(study, 'study_{}.pkl'.format(BOOSTING))
    best_params = study.best_params

else:

    best_params = {'num_leaves': 302,
 'max_depth': 157,
 'n_estimators': 1200,
 'subsample_for_bin': 290858,
 'min_child_samples': 79,
 'reg_alpha': 0.9919573524807885,
 'colsample_bytree': 0.5653288564015742,
 'learning_rate': 0.028565794309535042}

# %%
model.set_params(**best_params)

# %%
cross_val_score_auc(model,
                    X_train,
                    y_train,
                    n_fold=N_FOLD,
                    stratify=True,
                    shuffle=True,
                    random_state=RANDOM_STATE,
                    predict=True,
                    X_test=X_test,
                    submission=sample_submission)

# %%
ROC accuracy: 0.9760007886956308, Train: 0.9999998170917167
ROC accuracy: 0.978816647314765, Train: 0.9999998770998315
ROC accuracy: 0.9776080114780432, Train: 0.999999836281003
ROC accuracy: 0.9778474951671876, Train: 0.9999996537054818
ROC accuracy: 0.9758758593511995, Train: 0.9999998637893439
ROC accuracy: 0.9769108286745452, Train: 0.9999998926289491
ROC accuracy: 0.9785865478050013, Train: 0.9999998615712069
ROC accuracy: 0.9774573102118551, Train: 0.9999998197641444

# 0.9773879360872784
