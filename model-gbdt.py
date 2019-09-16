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

from codes.utils import import_data, cross_val_score_auc, reduce_mem_usage, fix_dtypes
from codes.fe_browser import latest
from codes.fe_emails import proton, mappings
from codes.fe_cards import stats
from codes.fe_date import dates
from codes.fe_relatives import divisions, divisions_float
from codes.fe_categorical import pairs, wtf, cat_limit, encode_cat
from codes.prepro import prepro
from codes.fe_users import users_stats

from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin

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
class Counter(TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        print(X.shape[1])
        return X


# %%
model = LGBMClassifier(metric='auc', 
                   boosting_type=BOOSTING)


# %%
prun = PrunedCV(N_FOLD, 0.02, minimize=False)


# %%
def objective(trial):
    
    joblib.dump(study, 'study_{}.pkl'.format(BOOSTING)) 

    
    params = {
        'selectfrommodel__threshold': trial.suggest_int('selectfrommodel__threshold', 1, 200),
        'lgbmclassifier__num_leaves': trial.suggest_int('lgbmclassifier__num_leaves', 10, 1500), 
        'lgbmclassifier__subsample_for_bin': trial.suggest_int('lgbmclassifier__subsample_for_bin', 1000, 5000000), 
        'lgbmclassifier__min_child_samples': trial.suggest_int('lgbmclassifier__min_child_samples', 200, 100000), 
        'lgbmclassifier__reg_alpha': trial.suggest_loguniform('lgbmclassifier__reg_alpha', 0.00000000001, 10.0),
        'lgbmclassifier__colsample_bytree': trial.suggest_loguniform('lgbmclassifier__colsample_bytree', 0.0001, 1.0),
        'lgbmclassifier__learning_rate': trial.suggest_loguniform('lgbmclassifier__learning_rate', 0.00001, 2.0)
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

    study.optimize(objective, timeout=60 * 60 * 13)
    joblib.dump(study, 'study_{}.pkl'.format(BOOSTING))
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
model.set_params(**best_params)

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
# ROC accuracy: 0.9668942182909179, Train: 0.9999901167411397
# ROC accuracy: 0.9720552290202384, Train: 0.9999891233350843
# ROC accuracy: 0.9710663975253696, Train: 0.9999918268060299
# ROC accuracy: 0.9703005116766165, Train: 0.9999910116495871
# ROC accuracy: 0.9677524410936837, Train: 0.9999883123936292
# ROC accuracy: 0.970521434805755, Train: 0.9999753389326952
# ROC accuracy: 0.9709850608667766, Train: 0.9999787304381259
# ROC accuracy: 0.9708245135815027, Train: 0.9999796449943333


# 0.9700499758576075

# %%
