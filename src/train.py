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

from . import dispatcher

TRAINING_DATA = os.environ.get('TRAINING_DATA')
TEST_DATA = os.environ.get('TEST_DATA')
FOLD = int(os.environ.get('FOLD')) # Fold must be an int
MODEL = os.environ.get('MODEL')

FOLD_MAPPING = {
    0:[1,2,3,4],
    1:[0,2,3,4],
    2:[0,1,3,4],
    3:[0,1,2,4],
    4:[0,1,2,3]
}

if __name__ == '__main__':
    # Load datasets
    df = pd.read_csv(TRAINING_DATA)

    print(len(df.columns))
    print(len(df.columns))
    ########## Partition ############### 
    # create train and validation sets from dictionary
    train = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))].reset_index(drop = True)
    valid = df[df.kfold==FOLD].reset_index(drop = True)

    #drop id variable from train
    # train = train.drop('id', axis = 1)
    # valid = valid.drop('id', axis = 1)

    # separate target and training data
    ytrain = train['target'].values
    yvalid = valid['target'].values
    train_df = train.drop(['target', 'kfold'], axis = 1)
    valid_df = valid.drop(['target', 'kfold'], axis = 1)

    # Make sure valid and train have the same order of columns
    valid_df = valid_df[train_df.columns]

    # Initialize model
    if MODEL == 'lightGBM':
        train_data = lgb.Dataset(train_df, ytrain)
        valid_data = lgb.Dataset(valid_df, yvalid, reference=train_data)

        # Initialize model
        model = lgb.LGBMClassifier(**{
                    'learning_rate': 0.01,
                    'feature_fraction': 0.1,
                    'min_data_in_leaf' : 15,
                    'max_depth': 5,
                    'reg_alpha': 1,
                    'reg_lambda': 1,
                    'objective': 'binary',
                    'metric': 'auc',
                    'n_jobs': -1,
                    'n_estimators' : 5000,
                    'feature_fraction_seed': 42,
                    'bagging_seed': 42,
                    'boosting_type': 'gbdt',
                    'verbose': 1,
                    'is_unbalance': True,
                    'boost_from_average': False})

        # Initialize model
        clf = model.fit(train_df, ytrain,
                          eval_set = [(train_df, ytrain), 
                                      (valid_df, yvalid)],
                          verbose = 1,
                          eval_metric = 'auc',
                          early_stopping_rounds = 200)
        preds = clf.predict_proba(valid_df)[:,1]
        print(metrics.roc_auc_score(yvalid, preds))

    elif MODEL == 'xgb':
        clf = dispatcher.MODELS[MODEL]
        clf.fit(train_df, ytrain,
          eval_set = [(train_df, ytrain), 
                    (valid_df, yvalid)],
            eval_metric = 'auc', 
          early_stopping_rounds=200, verbose = 100)
        preds = clf.predict_proba(valid_df)[:,1]
        print(metrics.roc_auc_score(yvalid, preds))

    else:
        clf = dispatcher.MODELS[MODEL]
        clf.fit(train_df, ytrain)
        preds = clf.predict_proba(valid_df)[:,1]
        print(metrics.roc_auc_score(yvalid, preds))

    ## Save Model for prediction
    joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")
    joblib.dump(metrics.roc_auc_score(yvalid, preds), f"fold_outputs/{MODEL}_{FOLD}_score.pkl")



    ## Feature importances
    # features = train.columns
    # importances = clf.feature_importances_
    # indices = np.argsort(importances)

    # fig, axs = plt.subplots(figsize = (15,16))
    # plt.title('Feature Importances')
    # plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    # plt.yticks(range(len(indices)), [features[i] for i in indices])
    # plt.xlabel('Relative Importance')
    # plt.show()
