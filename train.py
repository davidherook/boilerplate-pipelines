import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

data_path = 'data/avocado_v1.csv'

features = [
    'Total Volume', 
    '4046', 
    '4225', 
    '4770',
    'Total Bags', 
    'Small Bags', 
    'Large Bags',
    'XLarge Bags',
    'type',
    'year',
    'region'
]

numeric_features = ['Total Volume', 
    '4046', 
    '4225', 
    '4770',
    'Total Bags', 
    'Small Bags', 
    'Large Bags',
    'XLarge Bags'
]

categorical_features = ['type',
    'year',
    'region']

target = 'AveragePrice'

if __name__ == '__main__':

    df = pd.read_csv(data_path)
    print('Dataset: {}'.format(df.shape))

    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)
    print('Training on {}, testing on {}...'.format(len(X_train), len(X_test)))

    # ==================================================================
    # Show the difference in the datasets before, after transformation
    # ==================================================================
    print('\n' + '*'*40)
    print('\nBefore Transformations:')
    print('{}'.format(X_train.shape))
    print(X_train)

    numeric_transformer = Pipeline(steps = [
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps = [
        ('imputer', SimpleImputer(strategy='constant')),
        ('one hot', OneHotEncoder(handle_unknown='error'))
    ])

    transformer = ColumnTransformer(
        [('numeric_transformer', numeric_transformer, numeric_features),
         ('categorical_transformer', categorical_transformer, categorical_features)]
    )

    X_train_transf = transformer.fit_transform(X_train)

    print('\nAfter Transformations:')
    print('{}'.format(X_train_transf.shape))
    print(pd.DataFrame(X_train_transf))
    print('*'*40 + '\n')

    # ==================================================================
    # Create a master pipeline and fit the model
    # ==================================================================
    pipe = Pipeline(steps=[
        ('transformer', transformer),
        ('regressor', LinearRegression())
    ])

    model = pipe.fit(X_train, y_train)
    pickle.dump(model, open('model/model.pk', 'wb'))

    print('Train R2: {}'.format(model.score(X_train, y_train)))
    print('Test R2: {}'.format(model.score(X_test, y_test)))

    # ==================================================================
    # Show that fitting the transformed set returns the same as pipeline
    # ==================================================================
    model1 = LinearRegression().fit(X_train_transf, y_train)
    print('Train R2: {}'.format(model1.score(X_train_transf, y_train)))
    print('Test R2: {}'.format(model1.score(transformer.transform(X_test), y_test)))



