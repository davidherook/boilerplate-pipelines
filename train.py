import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

features = [
    'Total Volume', 
    '4046', 
    '4225', 
    '4770',
    'Total Bags', 
    'Small Bags', 
    'Large Bags',
    'XLarge Bags',
    # 'type',
    # 'year',
    # 'region'
]

target = 'AveragePrice'

if __name__ == '__main__':

    df = pd.read_csv('data/avocado.csv')
    print('Dataset: {}'.format(df.shape))

    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)
    print('Training on {}, testing on {}...'.format(len(X_train), len(X_test)))

    pipe = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    model = pipe.fit(X_train, y_train)
    pickle.dump(model, open('model/model.pk', 'wb'))

    print('Train R2: {}'.format(model.score(X_train, y_train)))
    print('Test R2: {}'.format(model.score(X_test, y_test)))



