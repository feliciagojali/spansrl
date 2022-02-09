import sys
import json
from models import SRL
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras import backend as K
import numpy as np
from features.SRLData import SRLData

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
    dir = config['features_dir'] +'train_'
    if (not config['use_fasttext']):
        features_1 = np.load(dir +config['features_1'], mmap_mode='r')
    else :
        features_1 = np.load(dir +config['features_1.1'], mmap_mode='r')
    features_2 = np.load(dir+config['features_2'], mmap_mode='r')
    features_3 = np.load(dir+config['features_3'], mmap_mode='r')
    input = [features_1, features_2, features_3]
    out = np.load(dir + config['output'], mmap_mode='r')

    # Features loading
    # dir = config['features_dir'] +'val_'
    # if (not config['use_fasttext']):
    #     features_1 = np.load(dir +config['features_1'], mmap_mode='r')
    # else :
    #     features_1 = np.load(dir +config['features_1.1'], mmap_mode='r')
    # features_2 = np.load(dir+config['features_2'], mmap_mode='r')
    # features_3 = np.load(dir+config['features_3'], mmap_mode='r')
    # input_val = [features_1, features_2, features_3]
    # out_val = np.load(dir + config['output'], mmap_mode='r')

    # Training Parameters
    batch_size = config['batch_size']
    epochs = config['epochs']
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    initial_learning_rate = 0.00075
    if (config['use_decay']):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=100,
            decay_rate=0.999,
            staircase=True)
    else:
        lr_schedule = initial_learning_rate
    # Compiling, fitting and saving model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=tf.keras.losses.CategoricalCrossentropy())
    model.fit(input, out, batch_size=batch_size, epochs=epochs, callbacks=[callback])
    model.save('models/'+ config['model_path'])


    # Predicting, unload model
    # data = SRLData(config, emb=False)

    # model = load_model(config['model_path'])
    # pred, idx_pred, idx_arg = model.predict(input)
    # res = data.convert_result_to_readable(pred, idx_arg, idx_pred)
    # real = data.convert_result_to_readable(out)
    # data.evaluate(real, res)
    # with open('data/results/'+ config['model_path'].split('/')[1]+'.txt', 'w') as f:
    #     for item in res:
    #         f.write("%s\n" %str(item))

     

if __name__ == "__main__":
    main()