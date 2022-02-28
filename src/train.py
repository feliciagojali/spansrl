import sys
import json
import numpy as np
import tensorflow as tf
from models import SRL
from utils import eval_validation, load_data

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

    devices = tf.config.experimental.list_physical_devices("GPU")
    device_names = [d.name.split("e:")[1] for d in devices]
    config_taken = [x for x in range(8)]
    taken_gpu = []
    for i, device_name in enumerate(device_names):
        if i in config_taken:
            taken_gpu.append(device_name)
    print(f"Taken GPU: {taken_gpu}")
    strategy = tf.distribute.MirroredStrategy(devices=taken_gpu)

    # Features loading
    input, out = load_data(config, 'train')
    input_val, out_val = load_data(config, 'val')

    # Training Parameters
    batch_size = config['batch_size']
    epochs = config['epochs']
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    initial_learning_rate = config['learning_rate']
    
    with strategy.scope():
        if (config['use_decay']):
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=100,
                decay_rate=0.999,
                staircase=True)
        else:
            lr_schedule = initial_learning_rate
        
        model = SRL(config)

        #checkpoint

        bestCheckpoint = tf.keras.callbacks.ModelCheckpoint("models/best_model",
                                                save_best_only=True)
        lastCheckpoint = tf.keras.callbacks.ModelCheckpoint("models/last_checkpoint",
                                                save_best_only=np.False_)
        # Compiling, fitting and saving model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=tf.keras.losses.CategoricalCrossentropy())
    if (config['use_pruning']):
        model.fit(input, out, batch_size=batch_size, epochs=epochs, callbacks=[callback, bestCheckpoint, lastCheckpoint])
    else:
        model.fit(input, out, batch_size=batch_size, validation_data=(input_val, out_val), epochs=epochs, callbacks=[callback, bestCheckpoint, lastCheckpoint])
    model.save(config['model_path'])

    eval_validation(config)
    

     

if __name__ == "__main__":
    main()