import sys
import json
from models import SRL
import tensorflow as tf
from utils import split_into_batch, read_from_batch
from keras import backend as K
import numpy as np
from sklearn.model_selection import train_test_split

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

    # Initialize model
    model = SRL(config)

    # Features loading
    dir = config['features_dir'] + 'train_'
    if (not config['use_fasttext']):
        features_1 = np.load(dir +config['features_1'], mmap_mode='r')
    else :
        features_1 = np.load(dir +config['features_1.1'], mmap_mode='r')
    features_2 = np.load(dir+config['features_2'], mmap_mode='r')
    features_3 = np.load(dir+config['features_3'], mmap_mode='r')
    input = [features_1, features_2, features_3]
    out = np.load(dir + config['output'], mmap_mode='r')

    # Training Parameters
    batch_size = config['batch_size']
    epochs = config['epochs']
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

    # f1_train, f1_test,f11_train, f11_test, f2_train, f2_test, f3_train, f3_test, out_train, out_test = train_test_split(features_1, features_11, features_2, features_3, out,test_size=0.4,train_size=0.6)
    # f1_test, f1_val, f11_test, f11_val, f2_test, f2_val, f3_test, f3_val, out_test,out_val = train_test_split(f1_train, f11_train, f2_train,f3_train,out_train,test_size = 0.5,train_size =0.5)
    # type = {
    #     "train": [f1_train, f11_train, f2_train, f3_train, out_train],
    #     "test": [f1_test, f11_test, f2_test,f3_test, out_test],
    #     "val": [f1_val, f11_val, f2_val, f3_val, out_val]
    # }
    # filename = [config['features_1'], config['features_1.1'], config['features_2'],config['features_3'], config['output']]
    # for key, val in type.items():
    #     for typ, name in zip(val, filename):
    #         np.save(dir+str(key)+'_'+name, typ)
    #         print(len(typ))

    # Compiling, fitting and saving model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.CategoricalCrossentropy())
    model.fit(input, out, batch_size=batch_size, epochs=epochs, callbacks=[callback])
    model.save('models/default_fasttext_'+str(config['use_fasttext']))

if __name__ == "__main__":
    main()