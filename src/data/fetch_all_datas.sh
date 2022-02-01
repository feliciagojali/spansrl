#! /bin/bash

DATA_PATH="./data"
if [ ! -d $DATA_PATH ]; then
  mkdir -p $DATA_PATH
fi

cd $DATA_PATH

RAW_PATH="./raw"
PRETRAINED_PATH="./pretrained"
FEATURES_PATH="./features"

# if [ ! -d $RAW_PATH ]; then
#   mkdir -p $RAW_PATH
# fi

# cd $RAW_PATH
# gdown --id 18f-dInJJPbuqXhHUdWpIeJ3XaBoERlh_

# cd ".."

# if [ ! -d $PRETRAINED_PATH ]; then
#   mkdir -p $PRETRAINED_PATH
# fi

# cd $PRETRAINED_PATH

# echo "Downloading pretrained models and data"
# gdown --id 1vL8vyfJbaj3i91peTu738jq25N-yhsU3
# gdown --id 1MQjcRLBCJsdk3AyCBWfAkltzRTHhI9ED
# unzip word2vec_news.model.wv.vectors.zip
# rm word2vec_news.model.wv.vectors.zip

# cd ".."

if [ ! -d $FEATURES_PATH ]; then
  mkdir -p $FEATURES_PATH
fi

cd $FEATURES_PATH
gdown --id 1shIQL_TuJBMnOKL0ZtqQEFrp58E654rp
unzip batch1_ft
rm batch1_ft.zip