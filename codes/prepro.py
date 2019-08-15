from sklearn.preprocessing import LabelEncoder


def prepro(train, test):
    for c in train.columns:
        if train[c].dtype == 'float16' or train[c].dtype == 'float32' or train[c].dtype == 'float64':
            train[c].fillna(train[c].mean())
            test[c].fillna(train[c].mean())


    # Label Encoding
    features = list(train.columns)
    features.remove('isFraud')
    for f in features:
        if train[f].dtype == 'object' or test[f].dtype == 'object':
            # lbl = LabelEncoder()
            # lbl.fit(list(train[f].values) + list(test[f].values))
            # train[f] = lbl.transform(list(train[f].values))
            # test[f] = lbl.transform(list(test[f].values))

            map_dict = train[[f, 'isFraud']].groupby(f).mean().to_dict()['isFraud']
            train[f + '_tv'] = train[f].map(map_dict)
            test[f + '_tv'] = test[f].map(map_dict)

            train.drop(f, axis=1, inplace=True)
            test.drop(f, axis=1, inplace=True)

    # fill in -999 for categorical
    train = train.fillna(-999)
    test = test.fillna(-999)

    return train, test
