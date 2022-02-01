#! /bin/bash

DATA_PATH="./data"
if [ ! -d $DATA_PATH ]; then
  mkdir -p $DATA_PATH
fi

cd $DATA_PATH

RAW_PATH="./raw"

if [ ! -d $RAW_PATH ]; then
  mkdir -p $RAW_PATH
fi

cd $RAW_PATH

echo "Downloading pretrained models and data"
gdown --id 1vL8vyfJbaj3i91peTu738jq25N-yhsU3
gdown --id 1MQjcRLBCJsdk3AyCBWfAkltzRTHhI9ED
unzip word2vec_news.model.wv.vectors.zip
rm word2vec_news.model.wv.vectors.zip
gdown --id 18f-dInJJPbuqXhHUdWpIeJ3XaBoERlh_
