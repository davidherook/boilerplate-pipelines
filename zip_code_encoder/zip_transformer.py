import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from train import generate_zip_codes
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class ZipTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, digits=5):
        self.digits = digits

    def split_impute(self, val):
        return list(str(val))[:self.digits]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X)
        Xt = X.applymap(self.split_impute)
        Xt = pd.DataFrame([i for i in Xt.iloc[:, 0]])
        return Xt


if __name__ == '__main__':
    df = pd.DataFrame(data=generate_zip_codes(100), columns=['zip'])
    print(df)

    zipcode_transformer = Pipeline(steps = [
        ('zipper', ZipTransformer(digits=5)),
        ('one hot', OneHotEncoder(handle_unknown='ignore'))
    ])

    transformer = ColumnTransformer(
        [('zipcode_transformer', zipcode_transformer, 'zip')],
        sparse_threshold=0
    )

    df_t = pd.DataFrame(transformer.fit_transform(df))
    print(df_t)
    print(df_t.shape)

    df = pd.DataFrame(data=generate_zip_codes(100), columns=['zip'])
    print(df)
    print(pd.DataFrame(transformer.transform(df)))


