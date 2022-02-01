import sys
import json
from models import SRL
import tensorflow as tf

from keras import backend as K
import numpy as np
def print_default():
    print('Configurations name not found.. Using the default config..')

def main():
    default = 'default'

    if (len(sys.argv) == 1):
        print('Attach which configurations you want to use for the model!') 
        print_default()
        config = default
    else :
        config = sys.argv[1]

    filename = './configurations.json'
    f = open(filename)
    all_config = json.load(f)
    
    try:
        config = all_config[config]
    except:
        print_default()
        config = all_config[default]

    model = SRL(config)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.CategoricalCrossentropy())

    emb1 = np.ones((5,30, 200))
    emb2 = np.ones((5,30,100))
    char = np.ones((5,30, 52))
    output = np.zeros((5, 59, 410, 15))
    out = np.ones((5, 59, 410, 1 ))
    out = np.concatenate([output, out], axis=-1)
    input = [emb1, emb2, char]
    model.fit(input, out)
    out, pred_idx_mask, arg_idx_mask = model.predict(input)

if __name__ == "__main__":
    main()