import numpy as np
import pandas as pd
import gc
import joblib
import os
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from datetime import datetime
from tqdm import tqdm_notebook


def reduce_mem_usage(df, verbose=True):
    numeric_formats = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numeric_formats:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
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


def drop_columns(train, test):
    cols_to_drop = ['V300', 'V309', 'V111', 'C3', 'V124', 'V106', 'V125', 'V315', 'V134', 'V102', 'V123', 'V316',
                    'V113',
                    'V136', 'V305', 'V110', 'V299', 'V289', 'V286', 'V318', 'V103', 'V304', 'V116', 'V29', 'V284',
                    'V293',
                    'V137', 'V295', 'V301', 'V104', 'V311', 'V115', 'V109', 'V119', 'V321', 'V114', 'V133', 'V122',
                    'V319',
                    'V105', 'V112', 'V118', 'V117', 'V121', 'V108', 'V135', 'V320', 'V303', 'V297', 'V120']

    print('{} features are going to be dropped for being useless'.format(len(cols_to_drop)))

    return train.drop(cols_to_drop, axis=1), test.drop(cols_to_drop, axis=1)


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
