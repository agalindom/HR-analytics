#export TRAINING_DATA=input/KF_train.csv
export TEST_DATA=input/new_test.csv

export MODEL=$1 # command so one can input the variable to run

python -m src.predict