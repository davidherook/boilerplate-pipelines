from sklearn.base import BaseEstimator, TransformerMixin
from train import generate_zip_codes
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class ZipTransformer(BaseEstimator, TransformerMixin):

    zipcode_len = 5

    def __init__(self, feature):
        self.feature = feature

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert X.shape[1] == 1
        for i in range(self.zipcode_len):
            new_col = f'{self.feature}_{i}'
            X[new_col] = X[self.feature].apply(lambda x: x[i])
        return X.drop(self.feature, axis=1)


if __name__ == '__main__':
    df = pd.DataFrame(data=generate_zip_codes(100), columns=['zip'])
    print(df)

    zipcode_transformer = Pipeline(steps = [
        ('zipper', ZipTransformer('zip')),
        ('one hot', OneHotEncoder(handle_unknown='ignore'))
    ])

    transformer = ColumnTransformer(
        [('zipcode_transformer', zipcode_transformer, ['zip'])],
        sparse_threshold=0
    )

    df_t = pd.DataFrame(transformer.fit_transform(df))
    print(df_t)
    print(df_t.shape)

    df = pd.DataFrame(data=generate_zip_codes(100), columns=['zip'])
    print(df)
    print(pd.DataFrame(transformer.transform(df)))


