import pandas as pd
import numpy as np


def pairs(train, test):
    for feature in ['id_02__id_20',
                    'id_02__D8',
                    'D11__device_name',
                    'device_name__P_emaildomain',
                    'P_emaildomain__C2',
                    'card2__dist1',
                    'card1__card5',
                    'card2__id_20',
                    'card3__card5',
                    'P_emaildomain_bin__R_emaildomain_bin',
                    'P_emaildomain_suffix__R_emaildomain_suffix',
                    'card5__P_emaildomain',
                    'addr1__card1',
                    'addr2__card1',
                    'addr1__card4',
                    'addr1__card1',
                    'addr1__card1']:
        f1, f2 = feature.split('__')
        train[feature] = train[f1].astype(str) + '_' + train[f2].astype(str)
        test[feature] = test[f1].astype(str) + '_' + test[f2].astype(str)

    return train, test

# def pairs(train, test):
#     c_cols1 = list(train.select_dtypes('object').columns)
#
#     for col in ['P_emaildomain', 'R_emaildomain', 'id_30', 'id_31', 'DeviceInfo']:
#         c_cols1.remove(col)
#
#     c_cols2 = c_cols1[1:].copy()
#     cnt = 0
#     for col1 in tqdm_notebook(c_cols1):
#         for col2 in tqdm_notebook(c_cols2):
#             train[col1 + '__concat__' + col2] = train[col1].astype('str') + '__' + train[col2].astype('str')
#             test[col1 + '__concat__' + col2] = test[col1].astype('str') + '__' + test[col2].astype('str')
#             cnt += 1
#         c_cols2 = c_cols2[1:]
#     print(cnt)
#     return train, test


def cat_limit(train, test, feature, limit=0):

    test_cnt = test[[feature, 'TransactionDT']].groupby(feature).count()['TransactionDT']
    train_cnt = train[[feature, 'TransactionDT']].groupby(feature).count()['TransactionDT']

    counts = pd.merge(train_cnt, test_cnt, how='inner', left_index=True, right_index=True)

    limiter = list(counts[counts.iloc[:, 0] > limit].index)

    test.loc[~test.loc[:, feature].isin(limiter), feature] = 'unknown'
    train.loc[~train.loc[:, feature].isin(limiter), feature] = 'unknown'

    return train, test


def wtf(train, test):

    for feature in ['id_01', 'id_31', 'id_33', 'id_35']:
        # Count encoded separately for train and test
        train[feature + '_count_dist'] = train[feature].map(train[feature].value_counts(dropna=False))
        test[feature + '_count_dist'] = test[feature].map(test[feature].value_counts(dropna=False))

    category_features = ["ProductCD", "P_emaildomain", #fixme
                         "R_emaildomain", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "DeviceType",
                         "DeviceInfo",
                         "id_12",
                         "id_13", "id_14", "id_15", "id_16", "id_17", "id_18", "id_19", "id_20", "id_21", "id_22",
                         "id_23",
                         "id_24",
                         "id_25", "id_26", "id_27", "id_28", "id_29", "id_30", "id_32", "id_34", 'id_36'
                                                                                                 "id_37", "id_38"]
    for c in category_features:
        train[feature + '_count_full'] = train[feature].map(
            pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
        test[feature + '_count_full'] = test[feature].map(
            pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))

    return train, test


def encode_cat(train, test):
    cat_list = list(train.dtypes[train.dtypes == 'object'].index)
    cats_to_list = []
    for feat in cat_list:
        train, test = cat_limit(train, test, feat)
        cats_to_list.extend(list(train[feat].unique()))

    cats_to_list = list(set(cats_to_list))
    cats_to_list.remove('unknown')

    cats_to_dict = {}
    cats_to_dict['unknown'] = -999
    cnt = 1
    for feat in cats_to_list:
        cats_to_dict[feat] = cnt
        cnt += 1

    for feat in cat_list:
        train[feat] = train[feat].map(cats_to_dict)
        test[feat] = test[feat].map(cats_to_dict)
        # np.random.seed(seed=42)
        # train.loc[train.loc[:, feat] < 0, feat] = np.random.rand((train.loc[:, feat] < 0).sum()) - 1
        # test.loc[test.loc[:, feat] < 0, feat] = np.random.rand((test.loc[:, feat] < 0).sum()) - 1

    return train, test
