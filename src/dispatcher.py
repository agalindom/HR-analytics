from sklearn import ensemble
import lightgbm as lgb
from xgboost import XGBClassifier

MODELS = {
    'randomForest' : ensemble.RandomForestClassifier(criterion = 'gini', n_estimators = 2000,
                                    max_depth=5, max_features='sqrt',
                                    min_samples_leaf=15, min_samples_split=10, 
                                    random_state = 42, verbose = 0, oob_score=True,
                                    n_jobs = -1),
    'extraTrees' : ensemble.ExtraTreesClassifier(
        n_jobs=-1, verbose=2,n_estimators = 200),
    'gradientBoosting': ensemble.GradientBoostingClassifier(n_estimators = 500),
    'lightGBM': 'lightgbm',
    'xgb': XGBClassifier(learning_rate=0.05, n_estimators=5000, booster = 'gbtree',importance_type = 'weight', gamma = 0.2,
                        objective='binary:logistic', seed=42, max_depth = 5, n_jobs = -1)
}