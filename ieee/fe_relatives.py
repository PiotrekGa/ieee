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
