import sys
import json
import nltk
import tensorflow as tf
from features import SRLData
from nltk.tokenize import word_tokenize
from utils import eval_validation, eval_test
from tensorflow.keras.models import load_model

nltk.download('punkt')

config = sys.argv[1]
filename = './configurations.json'
f = open(filename)
all_config = json.load(f)



with tf.device('/gpu:4'):
    try:
        config = all_config[config]
    except:
        print_default()
        config = all_config['default']

    while(True):
        print('1. Input your own sentence')
        print('2. Input sentences from file')
        try:
            opt = int(input('Please choose one of the options: '))
        except:
            continue
        if opt == 1 or opt == 2:
            break

    if (opt == 1):
        print('Please input one sentences at a time and please end with `END` if already done')
        print('Example, you want to inference 3 sentences, therefore type `Sentence 4: END`')
        sentences = []
        while(True):
            sent = str(input('Sentence ' + str(len(sentences) + 1) +' : '))
            if sent != 'END':
                if (sent == ''):
                    continue
                tokenized = word_tokenize(sent)
                sentences.append(tokenized)
            else:
                break
        if (len(sentences) == 0):
            print('Please enter at least one sentences!')
            sys.exit()
    else:
        while(True):
            sent = str(input('Please input filepath that contain the sentences: '))
            try:
                f = open(sent)
            except:
                continue
            sentences = [word_tokenize(s) for s in f.readlines()]
            break

    srl_data = SRLData(config)
    print('Extracting features...')

    srl_data.extract_features(sentences)
    if config['use_fasttext']:
        input_feat = [srl_data.word_emb_ft, srl_data.word_emb_2, srl_data.char_input]
    else:
        input_feat = [srl_data.word_emb_w2v, srl_data.word_emb_2, srl_data.char_input]

    srl_model = load_model(config['model_path'])

    if (config['use_pruning']):
        pred, idx_pred, idx_arg = srl_model.predict(input_feat, batch_size=config['batch_size'])
        res =  srl_data.convert_result_to_readable(pred, idx_arg, idx_pred)
    else:
        pred = srl_model.predict(input_feat, batch_size=config['batch_size'])
        res =  srl_data.convert_result_to_readable(pred)
    
    with open('data/results/' + sys.argv[2], 'w+') as f:
        for result in res:
            f.write(str(result) + '\n')

    print('You can see the results in data/results/' + sys.argv[2])

