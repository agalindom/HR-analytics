import pandas as pd 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import os
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
import numpy as np

from . import dispatcher

TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")

def predict():
    df = pd.read_csv(TEST_DATA)
    test_idx = df["enrollee_id"].values
    df = df.drop('enrollee_id', axis = 1)
    predictions = None

    for FOLD in range(5):
        print(FOLD)
        df = pd.read_csv(TEST_DATA)
        
        # data is ready to train
        clf = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}.pkl"))
        cols = joblib.load(os.path.join('models',f"{MODEL}_{FOLD}_columns.pkl"))
        df = df[cols]
        print(df.shape)
        preds = clf.predict_proba(df)[:, 1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds
    
    predictions /= 5

    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=["enrollee_id", "target"])
    return sub

if __name__ == '__main__':
    submission = predict()
    submission['enrollee_id'] = submission['enrollee_id'].astype(int)
    submission.to_csv(f'output/{MODEL}.csv', index = False)
