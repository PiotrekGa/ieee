import numpy as np


def dates(train, test):

    train['Transaction_day_of_week'] = np.floor((train['TransactionDT'] / (3600 * 24) - 1) % 7)
    test['Transaction_day_of_week'] = np.floor((test['TransactionDT'] / (3600 * 24) - 1) % 7)

    train['Transaction_hour_of_day'] = np.floor(train['TransactionDT'] / 3600) % 24
    test['Transaction_hour_of_day'] = np.floor(test['TransactionDT'] / 3600) % 24

    train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(
        int)
    test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)

    return train, test
