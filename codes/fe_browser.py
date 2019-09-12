import numpy as np
import gc


def latest(train, test):
    a = np.zeros(train.shape[0])
    train["latest_browser"] = a
    a = np.zeros(test.shape[0])
    test["latest_browser"] = a

    def set_browser(df):
        df['latest_browser'] = 0
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

        df.loc[:, 'DeviceInfo'] = df.loc[:, 'DeviceInfo'].str.lower()
        df.loc[:, 'id_30'] = df.loc[:, 'id_30'].str.lower()
        df.loc[:, 'id_31'] = df.loc[:, 'id_31'].str.lower()

        df.loc[:, 'device_version'] = df['DeviceInfo'].str.split('/', expand=True)[1]

        df.loc[:, 'os_id_30'] = df['id_30'].str.split(' ', expand=True)[0]
        df.loc[:, 'version_id_30'] = df['id_30'].str.replace('[^0-9\_\.\s]+', '')

        df.loc[df['id_31'].str.contains('android ', na=False), 'browser_id_31'] = 'android'
        df.loc[df['id_31'].str.contains('ie', na=False), 'browser_id_31'] = 'ie'
        df.loc[df['id_31'].str.contains('chrome', na=False), 'browser_id_31'] = 'chrome'
        df.loc[df['id_31'].str.contains('safari', na=False), 'browser_id_31'] = 'safari'
        df.loc[df['id_31'].str.contains('edge', na=False), 'browser_id_31'] = 'edge'
        df.loc[df['id_31'].str.contains('firefox', na=False), 'browser_id_31'] = 'firefox'
        df.loc[df['id_31'].str.contains('samsung', na=False), 'browser_id_31'] = 'samsung'
        df.loc[df['id_31'].str.contains('opera', na=False), 'browser_id_31'] = 'opera'
        df.loc[df['id_31'].str.contains('facebook', na=False), 'browser_id_31'] = 'facebook'
        df.loc[df['id_31'].str.contains('google', na=False), 'browser_id_31'] = 'google'

        df.loc[:, 'version_id_31'] = 'a_' + df['id_31'].str.replace('[^0-9\.\s]+', '')

        df.loc[:, 'screen_width'] = df['id_33'].str.split('x', expand=True)[0].astype(np.float32)
        df.loc[:, 'screen_height'] = df['id_33'].str.split('x', expand=True)[1].astype(np.float32)
        df.loc[:, 'screen_ratio'] = df.loc[:, 'screen_width'] / (df.loc[:, 'screen_height'] + 1)

        df.loc[df['DeviceInfo'].str.contains('build', na=False), 'device_name'] = 'handset'
        df.loc[df['DeviceInfo'].str.startswith('g3', na=False), 'device_name'] = 'sony'
        df.loc[df['DeviceInfo'].str.contains('-l', na=False), 'device_name'] = 'huawei'
        df.loc[df['DeviceInfo'].str.contains('xt', na=False), 'device_name'] = 'sony'
        df.loc[df['DeviceInfo'].str.contains('sm', na=False), 'device_name'] = 'samsung'
        df.loc[df['DeviceInfo'].str.contains('gt-', na=False), 'device_name'] = 'samsung'
        df.loc[df['DeviceInfo'].str.contains('lg-', na=False), 'device_name'] = 'lg'
        df.loc[df['DeviceInfo'].str.contains('rv:', na=False), 'device_name'] = 'rv'
        df.loc[df['DeviceInfo'].str.contains('ale-', na=False), 'device_name'] = 'huawei'
        df.loc[df['DeviceInfo'].str.startswith('lg', na=False), 'device_name'] = 'lg'

        df.loc[df['DeviceInfo'].str.contains('moto', na=False), 'device_name'] = 'motorola'
        df.loc[df['DeviceInfo'].str.contains('huawei', na=False), 'device_name'] = 'huawei'

        df.loc[df['DeviceInfo'].str.contains('blade', na=False), 'device_name'] = 'zte'
        df.loc[df['DeviceInfo'].str.contains('linux', na=False), 'device_name'] = 'linux'

        df.loc[df['DeviceInfo'].str.contains('htc', na=False), 'device_name'] = 'htc'
        df.loc[df['DeviceInfo'].str.contains('asus', na=False), 'device_name'] = 'asus'
        df.loc[df['DeviceInfo'].str.contains('samsung', na=False), 'device_name'] = 'samsung'

        df.loc[df['DeviceInfo'].str.contains('windows', na=False), 'device_name'] = 'windows'
        df.loc[df['DeviceInfo'].str.contains('ios', na=False), 'device_name'] = 'ios'
        df.loc[df['DeviceInfo'].str.contains('macos', na=False), 'device_name'] = 'macos'
        df.loc[df['DeviceInfo'].str.contains('trident', na=False), 'device_name'] = 'trident'
        df.loc[df['DeviceInfo'].str.contains('hisense', na=False), 'device_name'] = 'hisense'
        df.loc[df['DeviceInfo'].str.contains('pixel', na=False), 'device_name'] = 'pixel'

        df.loc[df['DeviceInfo'].str.contains('redmi', na=False), 'device_name'] = 'redmi'
        df.loc[df['DeviceInfo'].str.contains('lenovo', na=False), 'device_name'] = 'lenovo'
        df.loc[df['DeviceInfo'].str.contains('nexus', na=False), 'device_name'] = 'nexus'

        df.loc[df['DeviceInfo'].str.contains('ilium', na=False), 'device_name'] = 'ilium'
        df.loc[df['DeviceInfo'].str.contains('android', na=False), 'device_name'] = 'android'

        df.loc[df['DeviceInfo'].str.contains('hi6210sft', na=False), 'device_name'] = 'huawei'
        df.loc[df['DeviceInfo'].str.contains('f3213', na=False), 'device_name'] = 'sony'
        df.loc[df['DeviceInfo'].str.contains('f3113', na=False), 'device_name'] = 'sony'

        df.loc[df['DeviceInfo'].str.contains('microsoft', na=False), 'device_name'] = 'windows'

        gc.collect()
        return df

    return set_browser(train), set_browser(test)


