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

from catboost import CatBoostClassifier
import optuna

from codes.utils import cross_val_score_auc, PrunedCV, seed_everything

# %%
SEARCH_PARAMS = True
N_FOLD = 8
BOOSTING = 'cat'
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
train_df = joblib.load('train.pkl')[['TransactionDT']]
train_df['DT_M'] = train_df['TransactionDT'].apply(lambda x: (START_DATE + timedelta(seconds = x)))
train_df['DT_M'] = (train_df['DT_M'].dt.year-2017)*12 + train_df['DT_M'].dt.month 
train_df = train_df[train_df['DT_M']<(train_df['DT_M'].max())]

X_train['DT_M'] = train_df['DT_M']
X_train['DT_M'].fillna(17, inplace=True)

# %%
seed_everything(RANDOM_STATE)
y_sampled = pd.concat([y_train[y_train == 1], y_train[y_train == 0].sample(frac=0.2)])
X_train_sampled = X_train.loc[y_sampled.index, :]
group_split_sampled = X_train_sampled.DT_M
X_train_sampled.drop('DT_M', axis=1, inplace=True)
X_train.drop('DT_M', axis=1, inplace=True)

# %% [markdown]
# ### Model and training

# %%
seed_everything(RANDOM_STATE)
model = CatBoostClassifier(silent=True, custom_loss=['AUC'])
prun = PrunedCV(N_FOLD, 0.01, splits_to_start_pruning=3, minimize=False)


# %%
def objective(trial):
    
    joblib.dump(study, 'study_{}.pkl'.format(BOOSTING)) 

    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 3000), 
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 1.0), 
        'od_wait': trial.suggest_int('od_wait', 100, 2000), 
        'depth': trial.suggest_int('depth', 5, 12),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 200)
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

    study.optimize(objective, timeout=60 * 60 * 12)
    joblib.dump(study, 'study_{}.pkl'.format(BOOSTING))
    best_params = study.best_params

else:

    best_params = {
    'loss_function': 'Logloss',
    'custom_loss':['AUC'],
    'logging_level':'Silent',
    'early_stopping_rounds' : 100
}

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
# joblib.dump(xx, 'catboost.pkl')
