import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from features import SRLData

def create_span(length, max_span_length):
    span_start = []
    span_end = []
    span_width = []
    for i in range(length):
        for j in range(max_span_length):
            start_idx = i
            end_idx = i+j

            if (end_idx >= length):
                break
            span_start.append(start_idx)
            span_end.append(end_idx)
            span_width.append(end_idx - start_idx + 1)

    # Shape span_start, span_end: [num_spans]
    return span_start, span_end, span_width

def split_train_test_val(features_1, features_11, features_2, features_3, out, sentences, config):
    f1_train, f1_test,f11_train, f11_test, f2_train, f2_test, f3_train, f3_test, out_train, out_test, sent_train, sent_test= train_test_split(features_1, features_11, features_2, features_3, out, sentences, test_size=0.4,train_size=0.6)
    f1_test, f1_val, f11_test, f11_val, f2_test, f2_val, f3_test, f3_val, out_test,out_val, sent_test, sent_val= train_test_split(f1_test, f11_test, f2_test,f3_test,out_test,sent_test, test_size = 0.5,train_size =0.5)
    type = {
        "train": [f1_train, f11_train, f2_train, f3_train, out_train, sent_train],
        "test": [f1_test, f11_test, f2_test,f3_test, out_test, sent_test],
        "val": [f1_val, f11_val, f2_val, f3_val, out_val, sent_val]
    }
    filename = [config['features_1'], config['features_1.1'], config['features_2'],config['features_3'], config['output'], 'sentences.npy']
    for key, val in type.items():
        for typ, name in zip(val, filename):
            np.save(dir+str(key)+'_'+name, typ)
            print(len(typ))

def eval_validation(config):
    # Features loading
    dir = config['features_dir'] +'val_'
    if (not config['use_fasttext']):
        features_1 = np.load(dir +config['features_1'], mmap_mode='r')
    else :
        features_1 = np.load(dir +config['features_1.1'], mmap_mode='r')
    features_2 = np.load(dir+config['features_2'], mmap_mode='r')
    features_3 = np.load(dir+config['features_3'], mmap_mode='r')
    input = [features_1, features_2, features_3]
    out = np.load(dir + config['output'], mmap_mode='r')

    # Predicting, unload model
    data = SRLData(config, emb=False)

    model = load_model(config['model_path'])
    if (config['use_pruning']):
        pred, idx_pred, idx_arg = model.predict(input)
        res =  data.convert_result_to_readable(pred, idx_arg, idx_pred)
    else:
        pred = model.predict(input)
        res = data.convert_result_to_readable(pred)
    real = data.convert_result_to_readable(out)
    data.evaluate(real, res)
    with open('data/results/'+ config['model_path'].split('/')[1]+'.txt', 'w') as f:
        for item in res:
            f.write("%s\n" %str(item))