import pandas as pd


def users_stats(train, test):

    print('start')
    ids = ['id_' + str(i).zfill(2) for i in range(1, 39)]
    ident_train = train.loc[:, ids]
    ident_train.dropna(inplace=True, how='all')

    ident_test = test.loc[:, ids]
    ident_test.dropna(inplace=True, how='all')

    ident = pd.concat([ident_train, ident_test])

    hashs = pd.util.hash_pandas_object(ident, index=False).astype('str')
    print('hashing finished')

    all_data = pd.concat([train, test])
    all_data['user_id'] = hashs
    all_data = all_data[~all_data.user_id.isna()]
    print('concat data finished')

    gr = all_data[['user_id',
                   # 'card1',
                   # 'card2',
                   # 'card3',
                   # 'card5',
                   'TransactionAmt',
                   'TransactionDT']].groupby('user_id')

    print('grouping finished')

    groups = pd.DataFrame(gr.count()['TransactionAmt'])
    groups.columns = ['user_TransactionAmt_cnt']

    groups['user_TransactionAmt_mean'] = gr.mean()['TransactionAmt']
    groups['user_TransactionAmt_std'] = gr.std()['TransactionAmt']
    # groups['user_TransactionAmt_min'] = gr.min()['TransactionAmt']
    # groups['user_TransactionAmt_max'] = gr.max()['TransactionAmt']
    #
    # groups['user_TransactionDT_min'] = gr.min()['TransactionDT']
    # groups['user_TransactionDT_max'] = gr.max()['TransactionDT']
    # groups['user_TransactionDT_diff'] = groups['user_TransactionDT_max'] - groups['user_TransactionDT_min']

    # groups['user_card1_mean'] = gr.mean()['card1']
    # groups['user_card1_std'] = gr.std()['card1']
    # groups['user_card1_min'] = gr.min()['card1']
    # groups['user_card1_max'] = gr.max()['card1']
    #
    # groups['user_card2_mean'] = gr.mean()['card2']
    # groups['user_card2_std'] = gr.std()['card2']
    # groups['user_card2_min'] = gr.min()['card2']
    # groups['user_card2_max'] = gr.max()['card2']
    #
    # groups['user_card3_mean'] = gr.mean()['card3']
    # groups['user_card3_std'] = gr.std()['card3']
    # groups['user_card3_min'] = gr.min()['card3']
    # groups['user_card3_max'] = gr.max()['card3']
    #
    # groups['user_card5_mean'] = gr.mean()['card5']
    # groups['user_card5_std'] = gr.std()['card5']
    # groups['user_card5_min'] = gr.min()['card5']
    # groups['user_card5_max'] = gr.max()['card5']

    print('stats calculated')

    all_data = all_data[['user_id']]

    all_data = pd.merge(all_data, groups, left_on='user_id', right_index=True)
    all_data.drop('user_id', axis=1, inplace=True)
    print('merged all data')

    train = pd.merge(train, all_data, how='left', left_index=True, right_index=True)
    test = pd.merge(test, all_data, how='left', left_index=True, right_index=True)

    return train, test
