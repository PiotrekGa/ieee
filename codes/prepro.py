from sklearn.preprocessing import LabelEncoder


def prepro(train, test):
    for c in train.columns:
        if train[c].dtype == 'float16' or train[c].dtype == 'float32' or train[c].dtype == 'float64':
            train[c].fillna(train[c].mean())
            test[c].fillna(train[c].mean())

    # fill in -999 for categorical
    train = train.fillna(-999)
    test = test.fillna(-999)
    # Label Encoding
    for f in train.columns:
        if train[f].dtype == 'object' or test[f].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(train[f].values) + list(test[f].values))
            train[f] = lbl.transform(list(train[f].values))
            test[f] = lbl.transform(list(test[f].values))

    return train, test
