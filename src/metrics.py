import pandas as pd 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import numpy as np 
import warnings
warnings.filterwarnings("ignore")
from sklearn import model_selection
from sklearn import ensemble 
import lightgbm as lgb 
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

MODEL = os.environ.get("MODEL")

scores = []
for FOLD in range(5):
    scores.append(joblib.load(os.path.join("fold_outputs", f"{MODEL}_{FOLD}_score.pkl")))

print('\n FOLDS AUC score:')
print(np.array(scores).mean())