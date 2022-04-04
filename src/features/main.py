import sys
import json
import numpy as np
import tensorflow as tf
from helper import SRLData, split_train_test_val

def main():
    filename = './configurations.json'
    f = open(filename)
    all_config = json.load(f)

 
    config = all_config['default']
    
    # srl_data = SRLData(config)

    train = np.load('data/features/train_full_output.npy')
    val = np.load('data/features/val_full_output.npy')
    test = np.load('data/features/test_full_output.npy')
    
    
    print(len(train))
    print(len(val))
    print(len(test))




    
if __name__ == "__main__":
    with tf.device('/gpu:4'):
        main()