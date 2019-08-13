import pandas as pd


def users_stats(train, test):

    ids = ['id_' + str(i).zfill(2) for i in range(1, 39)]
    ident_train = train.loc[:, ids]
    ident_train.dropna(inplace=True, how='all')

    ident_test = test.loc[:, ids]
    ident_test.dropna(inplace=True, how='all')

    ident = pd.concat([ident_train, ident_test])

    hashs = pd.util.hash_pandas_object(ident, index=False).astype('str')

    all_data = pd.concat([train, test])

    all_data['user_id'] = hashs

    all_data = all_data[~all_data.user_id.isna()]

    gr = all_data.groupby('user_id')

    groups = pd.DataFrame(gr.count()['TransactionDT'])

    groups.columns = ['user_trx_cnt']

    # groups['user_fraund_cnt'] = gr.count()['isFraud']
    # groups['user_fraund_sum'] = gr.sum()['isFraud']
    # groups['user_fraund_ratio'] = groups['user_fraund_sum'] / groups['user_fraund_cnt']
    groups['user_TransactionAmt_mean'] = gr.mean()['TransactionAmt']
    groups['user_TransactionAmt_std'] = gr.std()['TransactionAmt']

    all_data = all_data[['user_id']]

    all_data = pd.merge(all_data, groups, left_on='user_id', right_index=True)
    all_data.drop('user_id', axis=1, inplace=True)

    train = pd.merge(train, all_data, how='left', left_index=True, right_index=True)
    test = pd.merge(test, all_data, how='left', left_index=True, right_index=True)

    return train, test
