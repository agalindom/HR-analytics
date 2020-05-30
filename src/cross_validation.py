import pandas as pd
from sklearn import model_selection

'''
For regression another approach is that you split in such a way 
you keep the distribution of the values similar in each and
every fold
'''

'''
DIFFERENT KINDS OF ML PROBLEMS:
* binary classification
* multi-class classification
* multi-label classification
* single-column regression
* multi-column regression
* GroupKfold classification/regression
* holdout (lots of samples(millions), or time series data)
'''


class CrossValidation:
    def __init__(
                self, 
                df,
                target_cols,
                problem_type = None,
                num_folds=5,
                shuffle = True,
                random_state = 42,
                multilabel_delimeter = ',',
                groupfold = None
                ):
        self.dataframe = df
        self.target_cols = target_cols
        self.num_target = len(target_cols)
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.multilabel_delimeter = multilabel_delimeter
        self.groupfold = groupfold

        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)
        
        self.dataframe['kfold'] = -1

    def split(self):
        if self.problem_type in ['binary_classification', 'multi-class classification']:
            if self.num_target != 1:
                raise Exception('Invalid number of targets for this problem type')
            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique()
            if unique_values == 1:
                raise Exception('Only one unique value found!')
            elif unique_values > 1:
                kf = model_selection.StratifiedKFold(n_splits=self.num_folds, 
                                                     shuffle=False)

                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=self.dataframe[target].values)):
                    self.dataframe.loc[val_idx, 'kfold'] = fold
            
        elif self.problem_type in ['single_col_regression', 'multi_col_regression']:
            if self.num_target != 1 and self.problem_type == 'single_col_regression':
                raise Exception('Invalid number of targets for this problem type')
            if self.num_target < 2 and self.problem_type == 'multi_col_regression':
                raise Exception('Invalid number of targets for this problem type')
            target = self.target_cols[0]
            kf = model_selection.KFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe)):
                self.dataframe.loc[val_idx, 'kfold'] = fold
        
        elif self.problem_type.startswith('holdout_'):
            holdout_percentage = int(self.problem_type.split('_')[1])
            num_holdout_samples = int(len(self.dataframe) * holdout_percentage/100)
            self.dataframe.loc[:len(self.dataframe) - num_holdout_samples, 'kfold'] = 0
            self.dataframe.loc[len(self.dataframe) - num_holdout_samples:, 'kfold'] = 1
        
        elif self.problem_type == 'multilabel_classification':
            if self.num_target != 1:
                raise Exception('Invalid number of targets for this problem type ')
            targets = self.dataframe[self.target_cols[0]].apply(lambda x: len(str(x).split(self.multilabel_delimeter)))
            kf = model_selection.StratifiedKFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=targets)):
                self.dataframe.loc[val_idx, 'kfold'] = fold
        
        elif self.problem_type in ['groupfold_regression', 'groupfold_classification']:
            if self.num_target != 1:
                raise Exception('INvalid number of targets for this problem type')
            target = self.target_cols[0]
            kf = model_selection.GroupKFold(n_splits = self.num_folds)
            for fold, (train_idx, valid_idx) in enumerate(kf.split(X=self.dataframe, group = self.groupfold)):
                self.dataframe.loc[val_idx, 'kfold'] = fold
        else: 
            raise Exception('Problem type not understood')

        return self.dataframe

if __name__ == '__main__':
    df = pd.read_csv("../input/new_train.csv")
    cv = CrossValidation(df, target_cols=['target'], 
            problem_type = 'binary_classification')
    df_split = cv.split()
    df_split.to_csv('../input/KF_train.csv', index = False)
    print(df_split.head())
    print(df_split.kfold.value_counts())