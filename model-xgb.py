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
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from datetime import timedelta

from xgboost import XGBClassifier
import optuna

from codes.utils import cross_val_score_auc, PrunedCV, seed_everything

# %%
SEARCH_PARAMS = False
N_FOLD = 8
BOOSTING = 'xgb'
RANDOM_STATE = 42
START_DATE = datetime.strptime('2017-11-30', '%Y-%m-%d')
seed_everything(RANDOM_STATE)


# %%
y_train = joblib.load('y_train.pkl')
X_train = joblib.load('features_train.pkl')
X_test = joblib.load('features_test.pkl')
sample_submission = pd.read_csv('../input/sample_submission.csv', index_col=0)
group_split = X_train.DT_M

# %%
# train_df = joblib.load('train.pkl')[['TransactionDT']]
# train_df['DT_M'] = train_df['TransactionDT'].apply(lambda x: (START_DATE + timedelta(seconds = x)))
# train_df['DT_M'] = (train_df['DT_M'].dt.year-2017)*12 + train_df['DT_M'].dt.month 
# train_df = train_df[train_df['DT_M']<(train_df['DT_M'].max())]

# X_train['DT_M'] = train_df['DT_M']
# X_train['DT_M'].fillna(17, inplace=True)

# %%
seed_everything(RANDOM_STATE)
y_sampled = pd.concat([y_train[y_train == 1], y_train[y_train == 0].sample(frac=0.2)])
X_train_sampled = X_train.loc[y_sampled.index, :]
group_split_sampled = X_train_sampled.DT_M
X_train_sampled.drop('DT_M', axis=1, inplace=True)
X_train.drop('DT_M', axis=1, inplace=True)

# %%
study = joblib.load('study_{}.pkl'.format(BOOSTING))

# %% [markdown]
# ### Model and training

# %%
del X_train_sampled, y_sampled, group_split_sampled, group_split

# %%
seed_everything(RANDOM_STATE)
model = XGBClassifier(n_jobs=-1, random_state=RANDOM_STATE)
prun = PrunedCV(N_FOLD, 0.01, splits_to_start_pruning=3, minimize=False)


# %%
def objective(trial):
    
    joblib.dump(study, 'study_{}.pkl'.format(BOOSTING)) 

    
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 1000), 
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5),
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 500),
        'subsample': trial.suggest_uniform('subsample', 0.01, 0.99),
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000)
        
    }
    
    model.set_params(**params)
    print(params)
        
    return prun.cross_val_score(model, 
                                X_train_sampled, 
                                y_sampled, 
                                split_type='stratifiedkfold',
                                shuffle=True,
                                metric='auc',
                                verbose=1,
                                random_state=RANDOM_STATE)

# %%
if SEARCH_PARAMS:
    if os.path.isfile('study_{}.pkl'.format(BOOSTING)):
        study = joblib.load('study_{}.pkl'.format(BOOSTING))
    else:
        study = optuna.create_study()

    study.optimize(objective, timeout=60 * 60 * 24)
    joblib.dump(study, 'study_{}.pkl'.format(BOOSTING))
    best_params = study.best_params

else:

    best_params = {'max_depth': 792,
                     'learning_rate': 0.014588159840197071,
                     'min_child_weight': 5,
                     'subsample': 0.6824025156854334,
                     'n_estimators': 1128}

# %%
model.set_params(**best_params)

# %%
seed_everything(RANDOM_STATE)
xx = cross_val_score_auc(model,
                        X_train,
                        y_train,
                        n_fold=N_FOLD,
                        random_state=RANDOM_STATE,
                        predict=True,
                        X_test=X_test,
                        shuffle=True,
                        split_type='stratifiedkfold',
                        return_to_stack=True,
                        submission=sample_submission)

# %%
joblib.dump(xx, 'xgboost.pkl')
