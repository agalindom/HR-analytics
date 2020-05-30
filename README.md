# Overview
- The repo contains a machine learning infraestructure thta predicts whether a particular employee id likely to look for a new job. 

# Problem Statement
- A particular company that does training for people in data science domains, connects different companies with their graduate enrollees, but not all the enrollees for the training program are looking for a new job, so the idea was to come up with a solution and create a machine learning model that predicts the probability of whether a particular enrollee is actually looking for a new job and connect them with thewir client companies.

# Features

    enrollee_id: int64
    city: object
    city_development_index: float64
    gender: object
    relevent_experience: object
    enrolled_university: object
    education_level: object
    major_discipline: object
    experience: object
    company_size: object
    company_type: object
    last_new_job: object
    training_hours: int64
    target: int64 


# Models
- Models where created using a 5 fold cross validation strategy and AUC as metric
Random Forest: 0.6638
LightGBM AUC: 0.6631
XGBoost AUC: 0.6718

# Usage 
- run the bash file **runTrain.sh** as follows:
> sh runTrain.sh  `<model_name>`>

- You can see and add avalilable models at **src/dispatcher.py**