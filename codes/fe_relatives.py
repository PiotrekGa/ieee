import numpy as np
from tqdm import tqdm_notebook


def divisions(train, test):

    train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / \
                                            train.groupby(['card1'])['TransactionAmt'].transform('mean')
    train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / \
                                            train.groupby(['card4'])['TransactionAmt'].transform('mean')
    train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / \
                                           train.groupby(['card1'])['TransactionAmt'].transform('std')
    train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / \
                                           train.groupby(['card4'])['TransactionAmt'].transform('std')

    test['TransactionAmt_to_mean_card1'] = test['TransactionAmt'] / \
                                           test.groupby(['card1'])['TransactionAmt'].transform('mean')
    test['TransactionAmt_to_mean_card4'] = test['TransactionAmt'] / \
                                           test.groupby(['card4'])['TransactionAmt'].transform('mean')
    test['TransactionAmt_to_std_card1'] = test['TransactionAmt'] / \
                                          test.groupby(['card1'])['TransactionAmt'].transform('std')
    test['TransactionAmt_to_std_card4'] = test['TransactionAmt'] / \
                                          test.groupby(['card4'])['TransactionAmt'].transform('std')

    train['id_02_to_mean_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('mean')
    train['id_02_to_mean_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('mean')
    train['id_02_to_std_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('std')
    train['id_02_to_std_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('std')

    test['id_02_to_mean_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('mean')
    test['id_02_to_mean_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('mean')
    test['id_02_to_std_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('std')
    test['id_02_to_std_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('std')

    train['D15_to_mean_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('mean')
    train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')
    train['D15_to_std_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('std')
    train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')

    test['D15_to_mean_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('mean')
    test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')
    test['D15_to_std_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('std')
    test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')

    train['D15_to_mean_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('mean')
    train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')
    train['D15_to_std_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('std')
    train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')

    test['D15_to_mean_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('mean')
    test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')
    test['D15_to_std_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('std')
    test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')

    return train, test


def divisions_float(X_train, X_test):

    print('aaa')

    columns = list(set(
    ['C{}'.format(i) for i in range(1,15)] \
    + ['D{}'.format(i) for i in range(1,16)] \
    + ['V' + str(i) for i in range(1,340)]))

    cols_to_check = columns.copy()

    CORR = 0.99

    cols1 = []
    for col in tqdm_notebook(columns):
        if col in X_train.columns:
            X_train[col + '_' + 'trx'] = X_train[col] / X_train.TransactionAmt
            X_test[col + '_' + 'trx'] = X_test[col] / X_test.TransactionAmt
            cols1.append(col + '_' + 'trx')
            cols_to_check.append(col + '_' + 'trx')

    X_corr = X_train.loc[:, cols1]
    print('calculating corrs 1 for ', X_corr.shape[1])
    corr_matrix = X_corr.corr(method='spearman').abs()
    del X_corr
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > CORR)]
    print('dropped because of high corr (', CORR, '): ', len(to_drop))
    X_train.drop(to_drop, axis=1, inplace=True)
    X_test.drop(to_drop, axis=1, inplace=True)

    for c in to_drop:
        cols_to_check.remove(c)

    c_cols1 = ['C{}'.format(i) for i in range(1,15)] + ['D{}'.format(i) for i in range(1,16)]
    c_cols2 = c_cols1[1:].copy()

    cols2 = []

    for col1 in tqdm_notebook(c_cols1):
        for col2 in c_cols2:
            X_train[col1 + '__div__' + col2] = X_train[col1] / X_train[col2]
            X_test[col1 + '__div__' + col2] = X_test[col1] / X_test[col2]
            cols2.append(col1 + '__div__' + col2)
            cols_to_check.append(col1 + '__div__' + col2)

        c_cols2 = c_cols2[1:]

    X_corr = X_train.loc[:, cols2]
    print('calculating corrs 2 for ', X_corr.shape[1])
    corr_matrix = X_corr.corr(method='spearman').abs()
    del X_corr
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > CORR)]
    print('dropped because of high corr (', CORR, '): ', len(to_drop))
    X_train.drop(to_drop, axis=1, inplace=True)
    X_test.drop(to_drop, axis=1, inplace=True)

    for c in to_drop:
        cols_to_check.remove(c)

    X_corr = X_train.loc[:, cols_to_check]
    print('calculating corrs 3 for ', X_corr.shape[1])
    corr_matrix = X_corr.corr(method='spearman').abs()
    del X_corr
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > CORR)]
    print('dropped because of high corr (', CORR, '): ', len(to_drop))
    X_train.drop(to_drop, axis=1, inplace=True)
    X_test.drop(to_drop, axis=1, inplace=True)

    return X_train, X_test
