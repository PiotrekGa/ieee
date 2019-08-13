from emails_dict import labels


def proton(train, test):

    train['P_isproton'] = (train['P_emaildomain'] == 'protonmail.com')
    train['R_isproton'] = (train['R_emaildomain'] == 'protonmail.com')
    test['P_isproton'] = (test['P_emaildomain'] == 'protonmail.com')
    test['R_isproton'] = (test['R_emaildomain'] == 'protonmail.com')

    return train, test


def mappings(train, test):

    emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other',
              'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',
              'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol',
              'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple',
              'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other',
              'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other',
              'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other',
              'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other', 'sc.rr.com': 'other',
              'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink',
              'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo',
              'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft',
              'mail.com': 'other', 'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other',
              'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other',
              'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol',
              'juno.com': 'other', 'icloud.com': 'apple'}
    us_emails = ['gmail', 'net', 'edu']
    for c in ['P_emaildomain', 'R_emaildomain']:
        train[c + '_bin'] = train[c].map(emails)
        test[c + '_bin'] = test[c].map(emails)

        train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])
        test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])

        train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
        test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

    return train, test


def labeling(train, test):

    for c1, c2 in train.dtypes.reset_index().values:
        if c2 == 'O':
            train[c1] = train[c1].map(lambda x: labels[str(x).lower()])
            test[c1] = test[c1].map(lambda x: labels[str(x).lower()])

    return train, test
