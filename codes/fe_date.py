import numpy as np


def dates(train, test):

    train['Transaction_day_of_week'] = np.floor((train['TransactionDT'] / (3600 * 24) - 1) % 7)
    test['Transaction_day_of_week'] = np.floor((test['TransactionDT'] / (3600 * 24) - 1) % 7)

    train['Transaction_hour_of_day'] = np.floor(train['TransactionDT'] / 3600) % 24
    test['Transaction_hour_of_day'] = np.floor(test['TransactionDT'] / 3600) % 24

    train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(
        int)
    test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)

    train = specials(train)
    test = specials(test)

    return train, test


def specials(df):
    df['special_hour_day'] = 0
    df.loc[df['Transaction_hour_of_day'] == 4, 'special_hour_day'] = 1
    df.loc[df['Transaction_hour_of_day'] == 5, 'special_hour_day'] = 1
    df.loc[df['Transaction_hour_of_day'] == 6, 'special_hour_day'] = 1
    df.loc[df['Transaction_hour_of_day'] == 7, 'special_hour_day'] = 1
    df.loc[df['Transaction_hour_of_day'] == 8, 'special_hour_day'] = 1
    df.loc[df['Transaction_hour_of_day'] == 9, 'special_hour_day'] = 1

    df['special_day_week'] = 0
    df.loc[df['Transaction_day_of_week'] == 3, 'special_day_week'] = 1
    df.loc[df['Transaction_day_of_week'] == 4, 'special_day_week'] = 1
    df.loc[df['Transaction_day_of_week'] == 5, 'special_day_week'] = 1
    df.loc[df['Transaction_day_of_week'] == 6, 'special_day_week'] = 1

    return df
