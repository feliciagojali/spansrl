#! /bin/bash

DATA_PATH="./data/features"
if [ ! -d $DATA_PATH ]; then
  mkdir -p $DATA_PATH
fi

MODEL_PATH="./models"
if [ ! -d $MODEL_PATH ]; then
  mkdir -p $MODEL_PATH
fi
