export TRAINING_DATA=input/KF_train.csv
#export TEST_DATA=input/new_test.csv

export MODEL=$1 # command so one can input the variable to run

FOLD=0 python -m src.train
FOLD=1 python -m src.train
FOLD=2 python -m src.train
FOLD=3 python -m src.train
FOLD=4 python -m src.train

python -m src.metrics