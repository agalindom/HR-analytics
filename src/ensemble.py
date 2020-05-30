import pandas as pd 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import os
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
import numpy as np

rf = pd.read_csv("../output/randomForest.csv")
lgb = pd.read_csv("../output/lightGBM.csv")
xgb = pd.read_csv("../output/xgb.csv")

print(rf.head())

submission = pd.read_csv("../input/sample_submission.csv")
submission['target'] = xgb["target"]*0.40 + rf["target"]*0.20 + lgb["target"]*0.20
submission.to_csv('../output/ensemble.csv', index=False)
print(submission.shape)
print(submission.sample(10))