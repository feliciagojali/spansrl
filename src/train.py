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
    features_1 = np.load(config['features_1'])
    features_2 = np.load(config['features_2'])
    features_3 = np.load(config['features_3'])
    batch_size = config['batch_size']
    epochs = config['epochs']

    out = np.load(config['train_out'])
    input = [features_1, features_2, features_3]
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.CategoricalCrossentropy())
    model.fit(input, out, batch_size=batch_size, epochs=epochs, callback=[callback])
    model.save('default')

if __name__ == "__main__":
    main()