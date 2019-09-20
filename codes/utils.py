import numpy as np
import pandas as pd
import gc
import joblib
import os
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from datetime import datetime
from tqdm import tqdm_notebook
from sklearn.base import TransformerMixin


def reduce_mem_usage(df, verbose=True):
    numeric_formats = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in tqdm_notebook(df.columns):
        col_type = df[col].dtypes
        if col_type in numeric_formats:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem,
                                                                              100 * (start_mem - end_mem) / start_mem))
    return df


def import_data(data_path):

    if os.path.isfile('train.pkl') & os.path.isfile('test.pkl'):

        train = joblib.load('train.pkl')
        test = joblib.load('test.pkl')

    else:

        train_transaction = pd.read_csv(data_path + 'train_transaction.csv', index_col='TransactionID')
        test_transaction = pd.read_csv(data_path + 'test_transaction.csv', index_col='TransactionID')
        train_identity = pd.read_csv(data_path + 'train_identity.csv', index_col='TransactionID')
        test_identity = pd.read_csv(data_path + 'test_identity.csv', index_col='TransactionID')

        train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
        del train_transaction, train_identity
        gc.collect()
        test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)
        del test_transaction, test_identity
        gc.collect()

        print('training set shape:', train.shape)
        print('test set shape:', test.shape)

        train = reduce_mem_usage(train)
        test = reduce_mem_usage(test)

        joblib.dump(train, 'train.pkl')
        joblib.dump(test, 'test.pkl')

    sample_submission = pd.read_csv(data_path + 'sample_submission.csv', index_col='TransactionID')

    return train, test, sample_submission


def cross_val_score_auc(model,
                        X_train,
                        y_train,
                        n_fold,
                        stratify=True,
                        shuffle=False,
                        random_state=None,
                        predict=False,
                        X_test=None,
                        submission=None,
                        verbose=1):

    if stratify:
        folds = StratifiedKFold(n_splits=n_fold, shuffle=shuffle, random_state=random_state)
    else:
        folds = KFold(n_splits=n_fold, shuffle=shuffle, random_state=random_state)

    model_scores = []
    for train_index, valid_index in tqdm_notebook(folds.split(X_train, y_train),
                                                  total=n_fold):

        if isinstance(X_train, np.ndarray):
            X_train_ = X_train[train_index]
            X_valid = X_train[valid_index]
        else:
            X_train_ = X_train.iloc[train_index, :]
            X_valid = X_train.iloc[valid_index, :]
        if isinstance(y_train, np.ndarray):
            y_train_ = y_train[train_index]
            y_valid = y_train[valid_index]
        else:
            y_train_ = y_train.iloc[train_index]
            y_valid = y_train.iloc[valid_index]

        model.fit(X_train_, y_train_)

        train_val = model.predict_proba(X_train_)[:, 1]
        val = model.predict_proba(X_valid)[:, 1]
        if predict:
            submission['isFraud'] = submission['isFraud'] + model.predict_proba(X_test)[:, 1] / n_fold
        del X_valid

        model_scores.append(roc_auc_score(y_valid, val))
        train_score = roc_auc_score(y_train_, train_val)

        if verbose > 0:
            print('ROC accuracy: {}, Train: {}'.format(model_scores[-1], train_score))
        del val, y_valid

        gc.collect()

    score = np.mean(model_scores)

    if predict:
        model_score = np.round(score, 5)
        timestamp = str(int(datetime.timestamp(datetime.now())))
        submission.to_csv('{}_submission_{}.csv'.format(timestamp, str(model_score)))

    print('')
    return score


def fix_dtypes(train, test):

    cols_to_object = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2'] + \
        ['id_' + str(i).zfill(2) for i in range(12, 39)]

    for col in cols_to_object:
        train[col] = train[col].astype('object')
        test[col] = test[col].astype('object')

    return train, test


class TargetEncoder(TransformerMixin):

    def __init__(self, target_name):
        self.target_name = target_name
        self.encoders_dict = dict()

    def fit(self, x, y):

        x = pd.concat([x, y], axis=1)
        for col in x.columns[:-1]:
            gr = x.groupby(col).mean()[self.target_name]
            self.encoders_dict[col] = gr

        x.drop(self.target_name, axis=1, inplace=True)
        return self

    def transform(self, x):

        for col in self.encoders_dict.keys():
            x[col] = x[col].map(self.encoders_dict[col])
            mean_to_impute = x[col].mean()
            x[col].fillna(mean_to_impute, inplace=True)

        return x


class Selector(TransformerMixin):

    def __init__(self, columns=None, return_vector=True):
        self.columns = columns
        self.return_vector = return_vector

    def fit(self, x, y=None):
        return self

    def transform(self, x):

        if len(self.columns) == 1:
            if self.return_vector:
                return x[self.columns[0]]
            else:
                return pd.DataFrame({self.columns[0]: x[self.columns[0]]})
        else:
            return x.loc[:, self.columns]


class MakeNonUnique(TransformerMixin):

    def __init__(self):
        self.to_drop = []

    def fit(self, x, y=None):
        self.to_drop = list(pd.DataFrame(x).columns[pd.DataFrame(x).nunique() == 1])
        return self

    def transform(self, x):
        return pd.DataFrame(x).drop(self.to_drop, axis=1)


class FrequencyEncoder(TransformerMixin):

    def __init__(self):
        self.encoders_dict = dict()

    def fit(self, x, y=None):

        for col in x.columns[:-1]:
            gr = x.groupby(col).count().iloc[:,0]
            self.encoders_dict[col] = gr
        return self

    def transform(self, x):

        for col in self.encoders_dict.keys():
            x[col] = x[col].map(self.encoders_dict[col])
            x[col].fillna(0, inplace=True)

        return x