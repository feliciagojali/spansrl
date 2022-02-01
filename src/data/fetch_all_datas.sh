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

# if [ ! -d $FEATURES_PATH ]; then
#   mkdir -p $FEATURES_PATH
# fi

# cd $FEATURES_PATH
# gdown --id 1IA7sG2TfTJ4exIUSL5aameVsr7EK6Ccw
# unzip batch1
# rm batch1.zip

# cd ".."

if [ ! -d $PROCESSED_PATH ]; then
  mkdir -p $PROCESSED_PATH
fi

cd $PROCESSED_PATH

echo "Downloading processed data"
gdown --id 1MwGa6GQKQU5ETqKTB3DdCDfIetH7edOv
unzip batch1_processed.zip
rm batch1_processed.zip