import numpy as np


def fe_browser_latest(train, test):
    a = np.zeros(train.shape[0])
    train["latest_browser"] = a
    a = np.zeros(test.shape[0])
    test["latest_browser"] = a

    def set_browser(df):
        df.loc[df["id_31"] == "samsung browser 7.0", 'latest_browser'] = 1
        df.loc[df["id_31"] == "opera 53.0", 'latest_browser'] = 1
        df.loc[df["id_31"] == "mobile safari 10.0", 'latest_browser'] = 1
        df.loc[df["id_31"] == "google search application 49.0", 'latest_browser'] = 1
        df.loc[df["id_31"] == "firefox 60.0", 'latest_browser'] = 1
        df.loc[df["id_31"] == "edge 17.0", 'latest_browser'] = 1
        df.loc[df["id_31"] == "chrome 69.0", 'latest_browser'] = 1
        df.loc[df["id_31"] == "chrome 67.0 for android", 'latest_browser'] = 1
        df.loc[df["id_31"] == "chrome 63.0 for android", 'latest_browser'] = 1
        df.loc[df["id_31"] == "chrome 63.0 for ios", 'latest_browser'] = 1
        df.loc[df["id_31"] == "chrome 64.0", 'latest_browser'] = 1
        df.loc[df["id_31"] == "chrome 64.0 for android", 'latest_browser'] = 1
        df.loc[df["id_31"] == "chrome 64.0 for ios", 'latest_browser'] = 1
        df.loc[df["id_31"] == "chrome 65.0", 'latest_browser'] = 1
        df.loc[df["id_31"] == "chrome 65.0 for android", 'latest_browser'] = 1
        df.loc[df["id_31"] == "chrome 65.0 for ios", 'latest_browser'] = 1
        df.loc[df["id_31"] == "chrome 66.0", 'latest_browser'] = 1
        df.loc[df["id_31"] == "chrome 66.0 for android", 'latest_browser'] = 1
        df.loc[df["id_31"] == "chrome 66.0 for ios", 'latest_browser'] = 1
        return df

    return set_browser(train), set_browser(test)
