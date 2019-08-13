import pandas as pd


def stats(train, test):

    train['card1_count_full'] = train['card1'].map(pd.concat([train['card1'], test['card1']],
                                                             ignore_index=True).value_counts(dropna=False))
    test['card1_count_full'] = test['card1'].map(pd.concat([train['card1'], test['card1']],
                                                           ignore_index=True).value_counts(dropna=False))

    train['card2_count_full'] = train['card2'].map(pd.concat([train['card2'], test['card2']],
                                                             ignore_index=True).value_counts(dropna=False))
    test['card2_count_full'] = test['card2'].map(pd.concat([train['card2'], test['card2']],
                                                           ignore_index=True).value_counts(dropna=False))

    train['card3_count_full'] = train['card3'].map(pd.concat([train['card3'], test['card3']],
                                                             ignore_index=True).value_counts(dropna=False))
    test['card3_count_full'] = test['card3'].map(pd.concat([train['card3'], test['card3']],
                                                           ignore_index=True).value_counts(dropna=False))

    train['card4_count_full'] = train['card4'].map(pd.concat([train['card4'], test['card4']],
                                                             ignore_index=True).value_counts(dropna=False))
    test['card4_count_full'] = test['card4'].map(pd.concat([train['card4'], test['card4']],
                                                           ignore_index=True).value_counts(dropna=False))

    train['card5_count_full'] = train['card5'].map(pd.concat([train['card5'], test['card5']],
                                                             ignore_index=True).value_counts(dropna=False))
    test['card5_count_full'] = test['card5'].map(pd.concat([train['card5'], test['card5']],
                                                           ignore_index=True).value_counts(dropna=False))

    train['card6_count_full'] = train['card6'].map(pd.concat([train['card6'], test['card6']],
                                                             ignore_index=True).value_counts(dropna=False))
    test['card6_count_full'] = test['card6'].map(pd.concat([train['card6'], test['card6']],
                                                           ignore_index=True).value_counts(dropna=False))

    train['addr1_count_full'] = train['addr1'].map(pd.concat([train['addr1'], test['addr1']],
                                                             ignore_index=True).value_counts(dropna=False))
    test['addr1_count_full'] = test['addr1'].map(pd.concat([train['addr1'], test['addr1']],
                                                           ignore_index=True).value_counts(dropna=False))

    train['addr2_count_full'] = train['addr2'].map(pd.concat([train['addr2'], test['addr2']],
                                                             ignore_index=True).value_counts(dropna=False))
    test['addr2_count_full'] = test['addr2'].map(pd.concat([train['addr2'], test['addr2']],
                                                           ignore_index=True).value_counts(dropna=False))

    return train, test
