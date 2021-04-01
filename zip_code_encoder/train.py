# =============================================
# Extract features from a zip code
# Should return a feature per position per digit (50 features)
#
# 1. Create feature for each digit position:
#    zip        0  1  2  3  4
#   30210  ->   3  0  2  1  0
#
# 2. One-hot encode:
#   0  1  2  3  4      0  1  2  3  4  5 ... 49
#   9  0  2  1  0  ->  0  0  0  1  0  0 ...  0
# =============================================

import numpy as np
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def generate_zip_codes(n=10):
    zips = []
    for i in range(n):
        zips.append(''.join([str(d) for d in np.random.randint(0,10,5)]))
    return zips
    
def create_cols(df, col):
    df1 = df.copy()
    max_value_len = 5
    new_cols = []
    for i in range(max_value_len):
        new_col = f'{col}_{i}'
        df1[new_col] = df1[col].apply(lambda x: x[i])
        new_cols.append(new_col)
    return df1, new_cols


if __name__ == '__main__':

    # ====================================
    # Transform using the create_cols function
    # ====================================
    df = pd.DataFrame(data=generate_zip_codes(100), columns=['zip'])
    df1, zipcode_features = create_cols(df, 'zip')

    zipcode_transformer0 = Pipeline(steps = [
        ('one hot', OneHotEncoder(handle_unknown='ignore'))
    ])

    transformer0 = ColumnTransformer(
        [('zipcode_transformer', zipcode_transformer0, zipcode_features)], 
        remainder='drop',
        sparse_threshold=0
    )

    df_t0 = pd.DataFrame(transformer0.fit_transform(df1))   # need to use df1

    # ====================================
    # Transform using custom transformer
    # ====================================
    from zip_transformer import ZipTransformer

    zipcode_transformer1 = Pipeline(steps = [
        ('zipper', ZipTransformer('zip')),
        ('one hot', OneHotEncoder(handle_unknown='ignore'))
    ])

    transformer1 = ColumnTransformer(
        [('zipcode_transformer', zipcode_transformer1, ['zip'])],
        sparse_threshold=0
    )

    df_t1 = pd.DataFrame(transformer1.fit_transform(df))   # use original df because transform is in pipeline

    assert df_t1.equals(df_t0)
    print(df.head(10))
    print(df_t1.head(10))

    # ====================================
    # Transform new zip data with fitted custom transformer
    # ====================================
    df2 = pd.DataFrame(data=generate_zip_codes(10), columns=['zip'])
    print(df2)
    print(pd.DataFrame(transformer1.transform(df2)))


