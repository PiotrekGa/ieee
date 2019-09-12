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

from codes.utils import import_data, drop_columns, cross_val_score_auc, reduce_mem_usage, fix_dtypes
from codes.fe_browser import latest
from codes.fe_emails import proton, mappings, labeling
from codes.fe_cards import stats
from codes.fe_date import dates
from codes.fe_relatives import divisions
from codes.fe_categorical import pairs, wtf, cat_limit
from codes.prepro import prepro
from codes.fe_users import users_stats

from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin

# %%
DATA_PATH = '../input/'
SEARCH_PARAMS = False
SEARCH_FEATURES = False
N_FOLD = 8


# %% {"_cell_guid": "79c7e3d0-c299-4dcb-8224-4455121ee9b0", "_uuid": "d629ff2d2480ee46fbb7e2d37f6b5fab8052498a"}
train, test, sample_submission = import_data(DATA_PATH)


# %% [markdown]
# ### Some Feature Engineering
#
# drop columns, count encoding, aggregation, fillna

# %%
train, test = fix_dtypes(train, test)

train, test = users_stats(train, test)

train, test = latest(train, test)

train, test = proton(train, test)

train['nulls1'] = train.isna().sum(axis=1)
test['nulls1'] = test.isna().sum(axis=1)

train, test = mappings(train, test)

train, test = stats(train, test)

train, test = divisions(train, test)

train, test = dates(train, test)

train, test = pairs(train, test)

cat_list = list(train.dtypes[train.dtypes == 'object'].index)
cats_to_list = []
for feat in cat_list:
    train, test = cat_limit(train, test, feat)
    cats_to_list.extend(list(train[feat].unique()))
    
cats_to_list = list(set(cats_to_list))
cats_to_list.remove('unknown')

cats_to_dict = {}
cats_to_dict['unknown'] = -1
cnt = 1 
for feat in cats_to_list:
    cats_to_dict[feat] = cnt
    cnt += 1
    
for feat in cat_list:
    train[feat] = train[feat].map(cats_to_dict)
    test[feat] = test[feat].map(cats_to_dict)
    train.loc[train.loc[:, feat] < 0, feat] = np.random.rand((train.loc[:, feat] < 0).sum()) - 1
    test.loc[test.loc[:, feat] < 0, feat] = np.random.rand((test.loc[:, feat] < 0).sum()) - 1

# train, test = wtf(train, test)

y_train = train['isFraud'].copy()

X_train = train.drop('isFraud', axis=1)
X_test = test.copy()

del train, test

#fill in mean for floats
X_train, X_test = prepro(X_train, X_test)

# %% [markdown]
# ### Model and training

# %%
columns = list(set(
['C{}'.format(i) for i in range(1,15)] \
+ ['D{}'.format(i) for i in range(1,16)] \
+ ['V' + str(i) for i in range(1,340)]))

for col in columns:
    if col in X_train.columns:
        X_train[col + '_' + 'trx'] = X_train[col] / X_train.TransactionAmt
        X_test[col + '_' + 'trx'] = X_test[col] / X_test.TransactionAmt

# %%
X_train = reduce_mem_usage(X_train)
X_test = reduce_mem_usage(X_test)

# %%
X_train[X_train == np.inf] = -1
X_train[X_train == -np.inf] = -1
X_test[X_test == np.inf] = -1
X_test[X_test == -np.inf] = -1
X_train[X_test.isna()] = -1
X_test[X_test.isna()] = -1

# %%
X_test.drop(['TransactionDT'], axis=1, inplace=True)
X_train.drop(['TransactionDT'], axis=1, inplace=True)

# %%
sfm = SelectFromModel(LGBMClassifier(metric='auc'), threshold=0.5)
sfm.fit(X_train, y_train)


print(X_train.shape[1])
columns = list(X_train.columns[sfm.get_support()])
print(len(columns))
X_train = X_train.loc[:,columns]
X_test = X_test.loc[:,columns]


# %%
class Counter(TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        print(X.shape[1])
        return X


# %%
model = make_pipeline(
    SelectFromModel(LGBMClassifier(metric='auc')),
#     Counter(),
    LGBMClassifier(metric='auc',
                   n_estimators=1000)
)

# %%
prun = PrunedCV(N_FOLD, 0.03, minimize=False)


# %%
def objective(trial):
    
    joblib.dump(study, 'study.pkl') 

    
    params = {
        'selectfrommodel__threshold': trial.suggest_int('selectfrommodel__threshold', 1, 100),
        'lgbmclassifier__num_leaves': trial.suggest_int('lgbmclassifier__num_leaves', 10, 1500), 
        'lgbmclassifier__subsample_for_bin': trial.suggest_int('lgbmclassifier__subsample_for_bin', 10, 3000000), 
        'lgbmclassifier__min_child_samples': trial.suggest_int('lgbmclassifier__min_child_samples', 2, 100000), 
        'lgbmclassifier__reg_alpha': trial.suggest_loguniform('lgbmclassifier__reg_alpha', 0.00000000001, 10.0),
        'lgbmclassifier__colsample_bytree': trial.suggest_loguniform('lgbmclassifier__colsample_bytree', 0.0001, 1.0),
        'lgbmclassifier__learning_rate': trial.suggest_loguniform('lgbmclassifier__learning_rate', 0.000001, 10.0)
    }
    
    print(params)
    
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

    study.optimize(objective, timeout=60 * 60 * 22)
    joblib.dump(study, 'study.pkl')
    best_params = study.best_params

else:

    best_params = {
        'selectfrommodel__threshold': 11,
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
                    random_state=42,
                    predict=True,
                    X_test=X_test,
                    submission=sample_submission)

# %%
# ROC accuracy: 0.9720145659778652, Train: 0.9999286851251716
# ROC accuracy: 0.9751973430652895, Train: 0.9999328083814578
# ROC accuracy: 0.9760593919549079, Train: 0.9998439324257781
# ROC accuracy: 0.9759738566311452, Train: 0.9999420044309371
# ROC accuracy: 0.972072614889331, Train: 0.9999141682082874
# ROC accuracy: 0.9726377525006845, Train: 0.9999124343601277
# ROC accuracy: 0.9749191757697562, Train: 0.9999151580131841
# ROC accuracy: 0.9739376183885246, Train: 0.9998564849540118


# 0.974101539897188
