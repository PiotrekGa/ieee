from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from lightgbm.sklearn import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


class RandomForestTransformer(RandomForestClassifier, TransformerMixin):

    def transform(self, X, *_):
        print('RandomForestTransformer')
        return self.predict(X).reshape(-1, 1)


class LGBMTransformer(LGBMClassifier, TransformerMixin):

    def transform(self, X, *_):
        print('LGBMTransformer')
        return self.predict(X).reshape(-1, 1)


class CatTransformer(CatBoostClassifier, TransformerMixin):

    def transform(self, X, *_):
        print('CatTransformer')
        return self.predict(X).reshape(-1, 1)


class XGBTransformer(XGBClassifier, TransformerMixin):

    def transform(self, X, *_):
        print('XGBTransformer')
        return self.predict(X).reshape(-1, 1)
