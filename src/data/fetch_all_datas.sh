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
gdown --id 11iUWswsf32A1fVtOEEYu6Q8muIGxI1EQ
gdown --id 11fqoZU3Pxhap8fwvuib03kXxxd6NU5UG
gdown --id 1-mXjuMTll3Hkn1vn-rzcywn37M7b_cD0
unzip val_features.zip
unzip train_features.zip
unzip test_features.zip

# gdown --id 1hXo67bbTyBBMYk1iJHam5rP9x7_Lq_mB
# unzip val_features.zip
# rm val_features.zip

cd ".."

if [ ! -d $RESULTS_PATH ]; then
  mkdir -p $RESULTS_PATH
fi
