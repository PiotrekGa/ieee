import numpy as np
import pandas as pd
import gc
import joblib
import os
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from tqdm import tqdm_notebook
from sklearn.base import TransformerMixin
from sklearn import metrics
import numpy
import pandas
import random


def sd(col, max_loss_limit=0.001, avg_loss_limit=0.001, na_loss_limit=0, n_uniq_loss_limit=0, fillna=0):
    """
    max_loss_limit - don't allow any float to lose precision more than this value. Any values are ok for GBT algorithms as long as you don't unique values.
                     See https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Precision_limitations_on_decimal_values_in_[0,_1]
    avg_loss_limit - same but calculates avg throughout the series.
    na_loss_limit - not really useful.
    n_uniq_loss_limit - very important parameter. If you have a float field with very high cardinality you can set this value to something like n_records * 0.01 in order to allow some field relaxing.
    """
    is_float = str(col.dtypes)[:5] == 'float'
    na_count = col.isna().sum()
    n_uniq = col.nunique(dropna=False)
    try_types = ['float16', 'float32']

    if na_count <= na_loss_limit:
        try_types = ['int8', 'int16', 'float16', 'int32', 'float32']

    for type in try_types:
        col_tmp = col

        # float to int conversion => try to round to minimize casting error
        if is_float and (str(type)[:3] == 'int'):
            col_tmp = col_tmp.copy().fillna(fillna).round()

        col_tmp = col_tmp.astype(type)
        max_loss = (col_tmp - col).abs().max()
        avg_loss = (col_tmp - col).abs().mean()
        na_loss = np.abs(na_count - col_tmp.isna().sum())
        n_uniq_loss = np.abs(n_uniq - col_tmp.nunique(dropna=False))

        if max_loss <= max_loss_limit and avg_loss <= avg_loss_limit and na_loss <= na_loss_limit and n_uniq_loss <= n_uniq_loss_limit:
            return col_tmp

    # field can't be converted
    return col


def reduce_mem_usage(df, deep=True, verbose=False, obj_to_cat=False):
    numerics = ['int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=deep).sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes

        # collect stats
        na_count = df[col].isna().sum()
        n_uniq = df[col].nunique(dropna=False)

        # numerics
        if col_type in numerics:
            df[col] = sd(df[col])

        # strings
        if (col_type == 'object') and obj_to_cat:
            df[col] = df[col].astype('category')

        if verbose:
            print(f'Column {col}: {col_type} -> {df[col].dtypes}, na_count={na_count}, n_uniq={n_uniq}')
        new_na_count = df[col].isna().sum()
        if na_count != new_na_count:
            print(
                f'Warning: column {col}, {col_type} -> {df[col].dtypes} lost na values. Before: {na_count}, after: {new_na_count}')
        new_n_uniq = df[col].nunique(dropna=False)
        if n_uniq != new_n_uniq:
            print(
                f'Warning: column {col}, {col_type} -> {df[col].dtypes} lost unique values. Before: {n_uniq}, after: {new_n_uniq}')

    end_mem = df.memory_usage(deep=deep).sum() / 1024 ** 2
    percent = 100 * (start_mem - end_mem) / start_mem
    if verbose:
        print('Mem. usage decreased from {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)'.format(start_mem, end_mem,
                                                                                              percent))
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


def cross_val_score_auc(models,
                        X_train,
                        y_train,
                        n_fold,
                        split_type='kfold',
                        shuffle=False,
                        random_state=None,
                        predict=False,
                        X_test=None,
                        submission=None,
                        stacking_type='lr',
                        verbose=1,
                        groups=None,
                        return_to_stack=False):
    if not isinstance(models, list):
        models = [models]

    if split_type == 'kfold':
        folds = KFold(n_splits=n_fold,
                      shuffle=shuffle,
                      random_state=random_state)

    elif split_type == 'stratifiedkfold':
        folds = KFold(n_splits=n_fold,
                      shuffle=shuffle,
                      random_state=random_state)

    elif split_type == 'groupkfold':
        folds = GroupKFold(n_splits=n_fold)

    model_scores = []

    if predict:
        submission['isFraud'] = 0

    if return_to_stack:
        stacking_data = []

    if split_type == 'groupkfold':
        kf = folds.split(X_train, y_train, groups)
    else:
        kf = folds.split(X_train, y_train)

    for train_index, valid_index in tqdm_notebook(kf, total=n_fold):

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

        for model in models:
            model.fit(X_train_, y_train_)

        if len(models) > 1:

            train_val_list = []
            vals_list = []
            if predict:
                subs_list = []
            for model in models:
                train_val_list.append(model.predict_proba(X_train_)[:, 1])
                vals_list.append(model.predict_proba(X_valid)[:, 1])
                if predict:
                    subs_list.append(model.predict_proba(X_test)[:, 1])

            if return_to_stack:
                stacking_data.append((train_val_list, vals_list, y_train_, y_valid))

            if stacking_type == 'mean':
                train_val = np.array(train_val_list).sum(axis=0) / len(models)
                val = np.array(vals_list).sum(axis=0) / len(models)
                if predict:
                    submission['isFraud'] += np.array(subs_list).sum(axis=0) / len(models)
            elif stacking_type == 'lr':
                stacker = LogisticRegression(solver='lbfgs', C=1 / 100)
                stacker.fit(np.array(train_val_list).T, y_train_)
                train_val = stacker.predict_proba(np.array(train_val_list).T)[:, 1]
                val = stacker.predict_proba(np.array(vals_list).T)[:, 1]
                if predict:
                    submission['isFraud'] += stacker.predict_proba(np.array(subs_list).T)[:, 1]
                if verbose > 0:
                    print(stacker.coef_)

            del train_val_list, vals_list

        else:
            train_val = models[0].predict_proba(X_train_)[:, 1]
            val = models[0].predict_proba(X_valid)[:, 1]
            if predict:
                pred = models[0].predict_proba(X_test)[:, 1]
                submission['isFraud'] = submission['isFraud'] + pred / n_fold

            if return_to_stack and predict:
                stacking_data.append((train_val, val, y_train_, y_valid, pred))
            elif return_to_stack:
                stacking_data.append((train_val, val, y_train_, y_valid))

        del X_valid

        model_scores.append(roc_auc_score(y_valid, val))
        train_score = roc_auc_score(y_train_, train_val)

        if verbose > 0:
            print('ROC accuracy: {}, Train: {}'.format(model_scores[-1], train_score))
        del val, y_valid

        gc.collect()

    score = np.mean(model_scores)

    if return_to_stack:
        return stacking_data

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


class Reporter(TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        print(x.shape[1])
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


def optimize_stacking(stack_data, params={'C': 0.1}):

    stacker = LogisticRegression(solver='lbfgs', **params)

    score = []
    for i in stack_data:
        x_train = np.array(i[0]).T
        x_val = np.array(i[1]).T
        y_train = i[2]
        y_val = i[3]
        stacker.fit(x_train, y_train)
        y_val_hat = stacker.predict_proba(x_val)
        score.append(roc_auc_score(y_val, y_val_hat[:, 1]))

    return sum(score) / len(score)


class PrunedCV:

    """PrunedCV applied pruning to cross-validation. Based on scores
    from initial splits (folds) is decides whether it's worth to
    continue the cross-validation. If not it stops the process and returns
    estimated final score.

    If the trial is worth checking (the initial scores are
    better than the best till the time or withing tolerance border) it's equivalent
    to standard cross-validation. Otherwise the trial is pruned.


    Args:
        cv:
            Number of folds to be created for cross-validation
        tolerance:
            Default = 0.1.
            The value creates boundary
            around the best score.
            If ongoing scores are outside the boundary,
            the trial is pruned.
        splits_to_start_pruning:
            Default = 2
            The fold at which pruning may be first applied.
        minimize:
            Default = True
            The direction of the optimization.

    Usage example:

        from lightgbm import LGBMRegressor
        from sklearn.datasets import fetch_california_housing
        from prunedcv import PrunedCV
        import numpy as np

        data = fetch_california_housing()
        x = data['data']
        y = data['target']

        pruner = PrunedCV(cv=8, tolerance=.1)

        model1 = LGBMRegressor(max_depth=25)
        model2 = LGBMRegressor(max_depth=10)
        model3 = LGBMRegressor(max_depth=2)

        pruner.cross_val_score(model1, x, y)
        pruner.cross_val_score(model2, x, y)
        pruner.cross_val_score(model3, x, y)

        print('best score: ', round(sum(pruner.best_splits_list_) / len(pruner.best_splits_list_),4))
            """

    def __init__(self,
                 cv,
                 tolerance=0.1,
                 splits_to_start_pruning=2,
                 minimize=True):

        if not isinstance(cv, int):
            raise TypeError
        if cv < 2:
            raise ValueError

        self.cv = cv
        self.set_tolerance(tolerance)
        self.splits_to_start_pruning = splits_to_start_pruning
        self.minimize = minimize
        self.prune = False
        self.cross_val_score_value = None
        self.current_splits_list_ = []
        self.best_splits_list_ = []
        self.first_run_ = True

    def set_tolerance(self,
                      tolerance):
        """Set tolerance value

        Args:
            tolerance:
            The value creates boundary
            around the best score.
            If ongoing scores are outside the boundary,
            the trial is pruned.
        """

        if not isinstance(tolerance, float):
            raise TypeError
        if tolerance < 0:
            raise ValueError

        self.tolerance = tolerance

    def cross_val_score(self,
                        model,
                        x,
                        y,
                        sample_weight=None,
                        split_type=None,
                        metric='mse',
                        shuffle=False,
                        random_state=None,
                        groups=None,
                        verbose=0):

        """Calculates pruned scores

        Args:
            model:
                An estimator to calculate cross-validated score
            x:
                numpy ndarray or pandas DataFrame
            y:
                numpy ndarray or pandas Series
            sample_weight:
                Default = None
                None or numpy ndarray or pandas Series
            metric:
                Default = 'mse'
                Metric from scikit-learn metrics to be optimized.
            shuffle:
                Default = False
                If True, shuffle the data before splitting them into folds.
            random_state:
                Default = None
                If any integer value, creates a seed for random number generation.

        Usage example:

            Check PrunedCV use example.
        """

        if not isinstance(x, (numpy.ndarray, pandas.core.frame.DataFrame)):
            raise TypeError

        if not isinstance(y, (numpy.ndarray, pandas.core.series.Series)):
            raise TypeError

        if metric not in ['mse',
                          'mae',
                          'accuracy',
                          'auc']:
            raise ValueError

        if split_type is None:

            if metric in ['mse',
                          'mae']:
                kf = KFold(n_splits=self.cv,
                           shuffle=shuffle,
                           random_state=random_state)

            elif metric in ['accuracy',
                            'auc']:

                kf = StratifiedKFold(n_splits=self.cv,
                                     shuffle=shuffle,
                                     random_state=random_state)

            else:
                raise ValueError

        else:

            if split_type == 'kfold':
                kf = KFold(n_splits=self.cv,
                           shuffle=shuffle,
                           random_state=random_state)

            elif split_type == 'stratifiedkfold':
                kf = StratifiedKFold(n_splits=self.cv,
                                     shuffle=shuffle,
                                     random_state=random_state)

            elif split_type == 'groupkfold':
                kf = GroupKFold(n_splits=self.cv)
            else:
                raise ValueError

        if split_type == 'groupkfold':
            kf_split = kf.split(x, y, groups)
        else:
            kf_split = kf.split(x, y)

        self.current_splits_list_ = []

        for train_idx, test_idx in kf_split:
            if not self.prune:

                if isinstance(x, numpy.ndarray):
                    x_train = x[train_idx]
                    x_test = x[test_idx]
                else:
                    x_train = x.iloc[train_idx, :]
                    x_test = x.iloc[test_idx, :]
                if isinstance(y, numpy.ndarray):
                    y_train = y[train_idx]
                    y_test = y[test_idx]
                else:
                    y_train = y.iloc[train_idx]
                    y_test = y.iloc[test_idx]

                if sample_weight is not None:
                    if isinstance(sample_weight, numpy.ndarray):
                        sample_weight_train = sample_weight[train_idx]
                        sample_weight_test = sample_weight[test_idx]
                    else:
                        sample_weight_train = sample_weight.iloc[train_idx]
                        sample_weight_test = sample_weight.iloc[test_idx]
                else:
                    sample_weight_train = None
                    sample_weight_test = None

                if sample_weight is not None:
                    model.fit(x_train, y_train, sample_weight=sample_weight_train)
                else:
                    model.fit(x_train, y_train)

                if metric == 'mse':
                    y_test_teor = model.predict(x_test)

                    self._add_split_value_and_prun(metrics.mean_squared_error(y_test,
                                                                              y_test_teor,
                                                                              sample_weight=sample_weight_test))

                elif metric == 'mae':
                    y_test_teor = model.predict(x_test)

                    self._add_split_value_and_prun(metrics.mean_absolute_error(y_test,
                                                                               y_test_teor,
                                                                               sample_weight=sample_weight_test))

                elif metric == 'accuracy':
                    y_test_teor = model.predict(x_test)

                    self._add_split_value_and_prun(metrics.accuracy_score(y_test,
                                                                          y_test_teor,
                                                                          sample_weight=sample_weight_test))

                elif metric == 'auc':
                    y_test_teor = model.predict_proba(x_test)[:, 1]

                    self._add_split_value_and_prun(metrics.roc_auc_score(y_test,
                                                                         y_test_teor,
                                                                         sample_weight=sample_weight_test))

                if verbose > 0:
                    print(self.current_splits_list_)
        if self.prune:
            print('pruned')
        self.prune = False
        return self.cross_val_score_value

    def _add_split_value_and_prun(self,
                                  value):

        if not isinstance(value, float):
            raise TypeError

        if len(self.current_splits_list_) == 0:
            self.prune = False

        if self.minimize:
            self.current_splits_list_.append(value)
        else:
            self.current_splits_list_.append(-value)

        if self.first_run_:
            self._populate_best_splits_list_at_first_run(value)
        else:
            self._decide_prune()

        if len(self.current_splits_list_) == self.cv:
            self._serve_last_split()

    def _populate_best_splits_list_at_first_run(self,
                                                value):

        if self.minimize:
            self.best_splits_list_.append(value)
        else:
            self.best_splits_list_.append(-value)

        if len(self.best_splits_list_) == self.cv:
            self.first_run_ = False

    def _decide_prune(self):

        split_num = len(self.current_splits_list_)
        mean_best_splits = sum(self.best_splits_list_[:split_num]) / split_num
        mean_curr_splits = sum(self.current_splits_list_) / split_num

        if self.cv > split_num >= self.splits_to_start_pruning:

            self.prune = self._significantly_higher_value(mean_best_splits,
                                                          mean_curr_splits,
                                                          self.minimize,
                                                          self.tolerance)

            if self.prune:
                self.cross_val_score_value = self._predict_pruned_score(mean_curr_splits,
                                                                        mean_best_splits)
                self.current_splits_list_ = []

    @staticmethod
    def _significantly_higher_value(mean_best_splits,
                                    mean_curr_splits,
                                    minimize,
                                    tolerance):
        tolerance_scaler_if_min = 1 + minimize * tolerance
        tolerance_scaler_if_max = 1 + (1 - minimize) * tolerance
        return mean_best_splits * tolerance_scaler_if_min < mean_curr_splits * tolerance_scaler_if_max

    def _predict_pruned_score(self,
                              mean_curr_splits,
                              mean_best_splits):
        return (mean_curr_splits / mean_best_splits) * (sum(self.best_splits_list_) / self.cv)

    def _serve_last_split(self):

        if sum(self.best_splits_list_) > sum(self.current_splits_list_):
            self.best_splits_list_ = self.current_splits_list_

        self.cross_val_score_value = sum(self.current_splits_list_) / self.cv
        self.current_splits_list_ = []


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)