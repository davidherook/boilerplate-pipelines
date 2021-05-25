# Use our trained model to make predictions on unseen data

import pickle
import pandas as pd
from train import features, target

df_path = 'data/avocado_test.csv'
model_path = 'model/model.pk'


if __name__ == '__main__':

    print('\n' + '*'*40)
    df = pd.read_csv(df_path)
    df_out_path = df_path.replace('.csv', '_predictions.csv')
    print('Predicting {} samples...'.format(df.shape[0]))

    model = pickle.load(open(model_path, 'rb'))

    X_test = df[features]
    df[target + '_pred'] = model.predict(X_test)
    df.to_csv(df_out_path, index=False)
    print(f'Predictions have been written to {df_out_path}')
    print('*'*40 + '\n')