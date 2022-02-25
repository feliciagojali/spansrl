#! /bin/bash

DATA_PATH="./data"
if [ ! -d $DATA_PATH ]; then
  mkdir -p $DATA_PATH
fi

cd $DATA_PATH

RAW_PATH="./raw"
PRETRAINED_PATH="./pretrained"
PROCESSED_PATH="./processed"
FEATURES_PATH="./features"

# if [ ! -d $RAW_PATH ]; then
#   mkdir -p $RAW_PATH
# fi

# cd $RAW_PATH
# gdown --id 18f-dInJJPbuqXhHUdWpIeJ3XaBoERlh_

# cd ".."

if [ ! -d $PRETRAINED_PATH ]; then
  mkdir -p $PRETRAINED_PATH
fi

cd $PRETRAINED_PATH

# echo "Downloading pretrained models and data"
# gdown --id 1vL8vyfJbaj3i91peTu738jq25N-yhsU3
# gdown --id 1MQjcRLBCJsdk3AyCBWfAkltzRTHhI9ED
# unzip word2vec_news.model.wv.vectors.zip
# rm word2vec_news.model.wv.vectors.zip

cd ".."

if [ ! -d $FEATURES_PATH ]; then
  mkdir -p $FEATURES_PATH
fi

cd $FEATURES_PATH

echo "Downloading features"
gdown --id 11iUWswsf32A1fVtOEEYu6Q8muIGxI1EQ
gdown --id 11fqoZU3Pxhap8fwvuib03kXxxd6NU5UG
unzip val_features.zip
unzip train_features.zip

# gdown --id 1hXo67bbTyBBMYk1iJHam5rP9x7_Lq_mB
# unzip val_features.zip
# rm val_features.zip

cd ".."

if [ ! -d $PROCESSED_PATH ]; then
  mkdir -p $PROCESSED_PATH
fi

cd $PROCESSED_PATH

echo "Downloading processed data"
# gdown --id 1MwGa6GQKQU5ETqKTB3DdCDfIetH7edOv
# unzip batch1_processed.zip
# rm batch1_processed.zip

# gdown --id 19D1dnT5vj-9Oh5n8LnWSJGyKGYX2mTCe
# unzip summary.zip
# rm summary.zip