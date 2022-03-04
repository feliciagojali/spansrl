import sys
import json
import tensorflow as tf
from utils import eval_validation, eval_test


config = sys.argv[1]
filename = './configurations.json'
f = open(filename)
all_config = json.load(f)

try:
    config = all_config[config]
except:
    print_default()
    config = all_config['validation']

print('Validating ' + str(sys.argv[2]) + ' data with model ' + config['model_path'])

with tf.device('/gpu:5'):
    if (sys.argv[2] == 'test'): 
        eval_test(config)
    else:
        eval_validation(config, False)
