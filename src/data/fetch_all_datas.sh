#! /bin/bash

DATA_PATH="./data"
RAW_PATH="./raw"
FEATURES_PATH="./features"
RESULTS_PATH = "./results"

cd $DATA_PATH

# if [ ! -d $RAW_PATH ]; then
#   mkdir -p $RAW_PATH
# fi

# cd $RAW_PATH
# gdown --id 18f-dInJJPbuqXhHUdWpIeJ3XaBoERlh_

# cd ".."


if [ ! -d $FEATURES_PATH ]; then
  mkdir -p $FEATURES_PATH
fi

cd $FEATURES_PATH

echo "Downloading features"
gdown --id 1-SmXIY2NuQLTLnl9qT0Abya3GMSHpilO
gdown --id 1cTc0An2FSXX-z3zod_UYA737NDh9gxR5
unzip val_features.zip
unzip train_features.zip

# gdown --id 1hXo67bbTyBBMYk1iJHam5rP9x7_Lq_mB
# unzip val_features.zip
# rm val_features.zip

cd ".."

if [ ! -d $RESULTS_PATH ]; then
  mkdir -p $RESULTS_PATH
fi
