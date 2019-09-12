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

from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin

# %%
DATA_PATH = '../input/'
SEARCH_PARAMS = True
SEARCH_FEATURES = False
N_FOLD = 8


# %% {"_cell_guid": "79c7e3d0-c299-4dcb-8224-4455121ee9b0", "_uuid": "d629ff2d2480ee46fbb7e2d37f6b5fab8052498a"}
train, test, sample_submission = import_data(DATA_PATH)


# %% [markdown]
# ### Some Feature Engineering
#
# drop columns, count encoding, aggregation, fillna

# %%
train, test = users_stats(train, test)

# train, test = drop_columns(train, test)

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
# ROC accuracy: 0.9707565062294428, Train: 0.9999416292415686
# ROC accuracy: 0.9758652343514882, Train: 0.9998818960438143
# ROC accuracy: 0.9747893539459415, Train: 0.9999033992474002
# ROC accuracy: 0.9741729952670382, Train: 0.999944229888998
# ROC accuracy: 0.9735064735460197, Train: 0.9999515715657177
# ROC accuracy: 0.9728703535857148, Train: 0.9999501665218518
# ROC accuracy: 0.9746020273044912, Train: 0.9999374155994768
# ROC accuracy: 0.973164729538134, Train: 0.9999402925194638


# 0.9737159592210338
