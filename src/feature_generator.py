import pandas as pd 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import numpy as np 
import matplotlib.pyplot as plt 
import itertools
import seaborn as sns 
sns.set(style="darkgrid")
import warnings
import statsmodels.api as sm
from tqdm import tqdm_notebook as tqdm
from sklearn import model_selection
from sklearn import preprocessing
import category_encoders as ce
warnings.filterwarnings("ignore")


####################################################################
############### Preprocessing functions ############################
####################################################################

def TargetEncoder(train, test, smooth):
    print('Target encoding...')
    train.sort_index(inplace=True)
    target = train['target']
    # test_id = test['enrollee_id']
    train.drop(['target'], axis=1, inplace=True)
    # test.drop(['enrollee_id'], axis=1, inplace=True)
    cat_feat_to_encode = train.columns.tolist()
    smoothing=smooth
    oof = pd.DataFrame([])
    for tr_idx, oof_idx in model_selection.StratifiedKFold(n_splits=5, random_state=42, shuffle=True).split(train, target):
        ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)
        ce_target_encoder.fit(train.iloc[tr_idx, :], target.iloc[tr_idx])
        oof = oof.append(ce_target_encoder.transform(train.iloc[oof_idx, :]), ignore_index=False)
    ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)
    ce_target_encoder.fit(train, target)
    train = oof.sort_index()
    print(test.shape)
    test = ce_target_encoder.transform(test)
    train['target'] = target
    features = list(train)
    print('Target encoding done!')
    return train, test,  features



def label_encode(data, feats):
    df = data.copy(deep = True)
    dict_of_dicts = {}
    for col in feats:
        length = []
        values = []
        encoder_dict = {}
        for idx, val in zip(range(len(df[col].unique())),df[col].unique()):
            if str(val) == 'nan':
                pass
            else:
                length.append(idx)
                values.append(val)
        for l,v in zip(length, values):
            encoder_dict[v] = l

        dict_of_dicts[col] = encoder_dict

        df[col] = df[col].replace(dict_of_dicts.get(col))
        
    return df

def CountEncoding(df, cols, df_test=None):
    for col in cols:
        frequencies = df[col].value_counts().reset_index()
        df_values = df[[col]].merge(frequencies, how='left', left_on=col, right_on='index').iloc[:,-1].values
        df[col+'_counts'] = df_values
        df_test_values = df_test[[col]].merge(frequencies, how='left', left_on=col, right_on='index').fillna(-1).iloc[:,-1].values
        df_test[col+'_counts'] = df_test_values
    count_cols = [col+'_counts' for col in cols]
    return df, df_test

def binary_processor(data):
    df = data.copy()
    
    # binary columns
    binary = ["relevent_experience"]
    
    #label encoder for binary feats
    df = label_encode(df, feats = binary)

    #sanity_check
    print('missing values for binary cols')
    for col in ['relevent_experience']:
        print(f'{col} null values: {df[col].isnull().sum()}/{df.shape[0]} - \
    prop: {round(df[col].isnull().sum()/df.shape[0], 3)}')
    
    return df

def full_nominal_processor(df, columns = None):
    df = df.copy()
    dummies = ["gender", "enrolled_university", "major_discipline", "company_type"]
    labeler = ["city"]

    df['gender'] = df['gender'].fillna('Unknown')
    df["enrolled_university"] = df["enrolled_university"].fillna('Unknown')
    df["major_discipline"] = df["major_discipline"].fillna('Unknown')
    df["company_type"] = df["company_type"].fillna('Unknown')

    df = pd.get_dummies(df, columns = dummies)
    df = label_encode(df, feats = labeler)

    tr = df[df['type'] == 'train']
    tt = df[df['type'] == 'test']

    tr, tt = CountEncoding(tr, labeler, df_test = tt)

    df = pd.concat([tr,tt])


    return df

def ordinal_processor(df):
    df = df.copy()
    # ordinal columns
    ordinal = ["education_level", "experience", "company_size", "last_new_job"]

    df["education_level"].fillna(-1, inplace = True)
    education_dict = {"Primary School": 0, "High School": 1, "Graduate":2, "Masters":3, "Phd": 4}

    df["experience"].fillna(-1, inplace = True)
    experience_dict = {'8':8, '9':9, '5':5, '14':14, '6':6, '1':1, '3':3, '13':13, '11':11, '7':7, '4':4,
                    '<1':0, '>20':21, '15':15, '12':12, '19':19, '18':18, '2':2, '20':20, '10':10, '17':17, '16':16}

    df["company_size"].fillna(-1, inplace = True)
    company_dict = {'<10':0, '100-500':3, '5000-9999':6, '500-999':4, '10/49':1, '1000-4999':5,
                    '50-99':2, '10000+':7}

    df["last_new_job"].fillna(-1, inplace = True)
    job_dict = {'never':0, '2':2, '>4':5, '4':4, '1':1, '3':3}

    dict_list = [education_dict, experience_dict, company_dict, job_dict]
    for idx, col in enumerate(ordinal):
        df[col] = df[col].replace(dict_list[idx])

    tr = df[df['type'] == 'train']
    tt = df[df['type'] == 'test']

    tr, tt = CountEncoding(tr, ordinal, df_test = tt)

    df = pd.concat([tr,tt])

    return df

def other_preprocessing(df):
    df = df.copy()
    df['training_hours'] = np.log(df['training_hours'])
    
    return df


if __name__ == '__main__':
    ## load the data
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')

    #join the data
    train['type'] = 'train'
    test['type'] = 'test'
    all_data = pd.concat([train,test])

    #imputation
    # cols_to_impute = []
    # for col in train.columns:
    #     if train[col].isnull().sum() > 0:
    #         cols_to_impute.append(col)

    # for col in cols_to_impute:
    #     all_data[col] = all_data[col].fillna(all_data[col].value_counts().index[0]) 

    all_data = binary_processor(all_data)
    all_data = full_nominal_processor(all_data)
    all_data = ordinal_processor(all_data)
    all_data = other_preprocessing(all_data)

    train = all_data[all_data['type'] == 'train']
    test = all_data[all_data['type'] == 'test']

    #drop type
    train= train.drop('type', axis = 1)
    train = train.iloc[:-4]
    test= test.drop(['type', 'target'], axis = 1)

    train, test = train.reset_index(drop = True), test.reset_index(drop = True)
    print(train.info())

    train = pd.get_dummies(train)
    test = pd.get_dummies(test)

    print(len(train.columns))
    print(train.columns)
    print(train.isnull().sum())
    # print(train.head())
    # print(train.shape)


    # print(test.columns)
    # print(train.columns)
    train = train.drop('enrollee_id', axis = 1)
    test_id = test["enrollee_id"]
    test = test.drop('enrollee_id', axis = 1)
    # train_, test_ = CountEncoding(train, list(train.columns), df_test = test)
    # cols_ = [i for i in train_.columns if '_count' in i]
    # train['target'] = target
    enc_cols_tr = ['city', 'city_development_index', 'company_size', 'education_level',
       'experience', 'last_new_job', 'relevent_experience',
       'target']
    enc_cols_tt = ['city', 'city_development_index', 'company_size', 'education_level',
       'experience', 'last_new_job', 'relevent_experience']
    train_, test_, features = TargetEncoder(train[enc_cols_tr], test[enc_cols_tt], 0.1)
    train[enc_cols_tr] = train_
    test[enc_cols_tt] = test_
    test["enrollee_id"] = test_id
    print(train.info())

    #save results
    train.to_csv('../input/new_train.csv', index = False)
    test.to_csv('../input/new_test.csv', index = False)

    
