from sklearn.preprocessing import LabelEncoder
import pandas as pd


def pairs(train, test):
    for feature in ['id_02__id_20', 'id_02__D8', 'D11__DeviceInfo', 'DeviceInfo__P_emaildomain', 'P_emaildomain__C2',
                    'card2__dist1', 'card1__card5', 'card2__id_20', 'card5__P_emaildomain', 'addr1__card1']:
        f1, f2 = feature.split('__')
        train[feature] = train[f1].astype(str) + '_' + train[f2].astype(str)
        test[feature] = test[f1].astype(str) + '_' + test[f2].astype(str)

        le = LabelEncoder()
        le.fit(list(train[feature].astype(str).values) + list(test[feature].astype(str).values))
        train[feature] = le.transform(list(train[feature].astype(str).values))
        test[feature] = le.transform(list(test[feature].astype(str).values))

    return train, test


def wtf(train, test):

    for feature in ['id_34', 'id_36']:
        # Count encoded for both train and test
        train[feature + '_count_full'] = train[feature].map(
            pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
        test[feature + '_count_full'] = test[feature].map(
            pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))

    for feature in ['id_01', 'id_31', 'id_33', 'id_35', 'id_36']:
        # Count encoded separately for train and test
        train[feature + '_count_dist'] = train[feature].map(train[feature].value_counts(dropna=False))
        test[feature + '_count_dist'] = test[feature].map(test[feature].value_counts(dropna=False))

    category_features = ["ProductCD", "P_emaildomain",
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
