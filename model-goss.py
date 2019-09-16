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
DATA_PATH = '../input/'
SEARCH_PARAMS = True
N_FOLD = 5
BOOSTING = 'goss'
RANDOM_STATE = 43


# %% {"_cell_guid": "79c7e3d0-c299-4dcb-8224-4455121ee9b0", "_uuid": "d629ff2d2480ee46fbb7e2d37f6b5fab8052498a"}
X_train, X_test, sample_submission = import_data(DATA_PATH)


# %% [markdown]
# ### Some Feature Engineering
#
# drop columns, count encoding, aggregation, fillna

# %%
if os.path.isfile('features_train.pkl'):
    X_train = joblib.load('features_train.pkl')
    X_test = joblib.load('features_test.pkl')
    y_train = joblib.load('y_train.pkl')
    
else:

    print('fix_dtypes')
    X_train, X_test = fix_dtypes(X_train, X_test)
    print('users_stats')
    X_train, X_test = users_stats(X_train, X_test)
    print('latest')
    X_train, X_test = latest(X_train, X_test)
    print('proton')
    X_train, X_test = proton(X_train, X_test)
    print('nulls1')
    X_train['nulls1'] = X_train.isna().sum(axis=1)
    X_test['nulls1'] = X_test.isna().sum(axis=1)
    print('mappings')
    X_train, X_test = mappings(X_train, X_test)
    print('stats')
    X_train, X_test = stats(X_train, X_test)
    print('divisions')
    X_train, X_test = divisions(X_train, X_test)
    print('dates')
    X_train, X_test = dates(X_train, X_test)
    print('pairs')
    X_train, X_test = pairs(X_train, X_test)
    print('encode_cat')
    X_train, X_test = encode_cat(X_train, X_test)
    print('wtf')
    # X_train, X_test = wtf(X_train, X_test)
    print('y')
    y_train = X_train['isFraud'].copy()
    X_train = X_train.drop('isFraud', axis=1)
    print('divisions_float')
    X_train, X_test = divisions_float(X_train, X_test)
    print('prepro')
    X_train, X_test = prepro(X_train, X_test)
    print('reduce_mem_usage')
    # X_train = reduce_mem_usage(X_train)
    # X_test = reduce_mem_usage(X_test)
    print('np.inf')
    X_train[X_train == np.inf] = -1
    X_train[X_train == -np.inf] = -1
    X_train[X_train.isna()] = -1
    X_test[X_test == np.inf] = -1
    X_test[X_test == -np.inf] = -1
    X_test[X_test.isna()] = -1
    print('TransactionDT')
    X_test.drop(['TransactionDT'], axis=1, inplace=True)
    X_train.drop(['TransactionDT'], axis=1, inplace=True)
    
    joblib.dump(X_train, 'features_train.pkl')
    joblib.dump(X_test, 'features_test.pkl')
    joblib.dump(y_train, 'y_train.pkl')

# %%
train_new_feats = joblib.load('train_feats.pkl')
test_new_feats = joblib.load('test_feats.pkl')

for col in train_new_feats.select_dtypes('category').columns:
    train_new_feats.loc[:, col] = train_new_feats.loc[:, col].astype('int')
    test_new_feats.loc[:, col] = test_new_feats.loc[:, col].astype('int')

print('np.inf')
train_new_feats[train_new_feats == np.inf] = -1
train_new_feats[train_new_feats == -np.inf] = -1
train_new_feats[train_new_feats.isna()] = -1
test_new_feats[test_new_feats == np.inf] = -1
test_new_feats[test_new_feats == -np.inf] = -1
test_new_feats[test_new_feats.isna()] = -1

print(train_new_feats.shape[1])
print(X_train.shape[1])
X_train = pd.concat([X_train, train_new_feats], axis=1)
print(X_train.shape[1])
X_test = pd.concat([X_test, test_new_feats], axis=1)
del train_new_feats, test_new_feats

# %% [markdown]
# ### Model and training

# %%
sel_mod = LGBMClassifier(metric='auc', boosting_type=BOOSTING)
sfm = SelectFromModel(sel_mod, threshold=0.5)
print(X_train.shape[1])
sfm.fit(X_train, y_train)
columns = list(X_train.columns[sfm.get_support()])
print(len(columns))
X_train = X_train.loc[:,columns]
X_test = X_test.loc[:,columns]


# %%
# 854
# 481

# %%
class Counter(TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        print(X.shape[1])
        return X


# %%
sel_mod = LGBMClassifier(metric='auc', n_estimators=200, boosting_type=BOOSTING)
sel_mod.set_params(**pars)
model = make_pipeline(
    SelectFromModel(sel_mod),
    Counter(),
    LGBMClassifier(metric='auc',
                   boosting_type=BOOSTING,
                   n_estimators=2000)
)

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

    study.optimize(objective, timeout=60 * 60 * 22)
    joblib.dump(study, 'study_{}.pkl'.format(BOOSTING))
    best_params = study.best_params

else:

    best_params = {
        'selectfrommodel__threshold': 20,
        'lgbmclassifier__num_leaves': 330,
        'lgbmclassifier__subsample_for_bin': 2077193,
        'lgbmclassifier__min_child_samples': 2227,
        'lgbmclassifier__reg_alpha': 0.16758905622425835,
        'lgbmclassifier__colsample_bytree': 0.49030006727392056,
        'lgbmclassifier__learning_rate': 0.07916040470631734
    }

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
