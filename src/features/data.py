
from bdb import effective
from imp import new_module
import os
import sys
from venv import create


currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from helper import split_first, label_encode, get_span_idx, pad_input, extract_bert
from utils.utils import create_span
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from gensim.models import fasttext
import torch 

unk_label = '<unk>'
pad_label = '<pad>'
train_data_path = "/data/raw/"

class SRLData(object):
    def __init__(self, config):
        self.config = config
        # Configurations and constants
        self.max_arg_span = config['max_arg_span']
        self.max_pred_span = config['max_pred_span']
        self.max_tokens = config['max_tokens']
        self.num_labels = config['num_srl_labels']
        self.max_char = config['max_char']
        self.arg_span_idx = create_span(self.max_tokens, self.max_arg_span)
        self.pred_span_idx = create_span(self.max_tokens, self.max_pred_span)

        # Initial Data
        self.sentences = []
        self.arg_list = []

        # Features Input

        ## Word Embedding
        self.padding_side = config['padding_side']
        self.word_vec = fasttext.load_facebook_vectors(config['word_emb_path'])
        self.word_emb = []

        self.bert_model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
        self.bert_tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1", padding_side=self.padding_side)
        self.bert_emb = []
        # Output
        self.output = []

        # Dict
        self.labels_mapping = label_encode(config['srl_labels'])


    # To convert train data labels to output models
    def convert_train_output(self):
        batch_size = len(self.sentences)
        pred_start, pred_end, _ = self.pred_span_idx
        arg_start, arg_end, _ = self.arg_span_idx

        num_preds = len(pred_start)
        num_args = len(arg_start)
        # Fill with null labels first
        # initialTensor = tf.fill([batch_size, num_preds, num_args, self.num_labels+1], len(self.labels_mapping) + 1)
        initialData = np.zeros([batch_size, num_preds, num_args, self.num_labels])
        initialLabel = np.ones([batch_size, num_preds, num_args, 1])
        initialData = np.concatenate([initialData, initialLabel], axis=-1)
        indices = []
        for idx_sent, sentences in enumerate(self.arg_list):
            for PAS in sentences:
                id_pred_span = get_span_idx(PAS['id_pred'], self.pred_span_idx)
                arg_list = PAS['args']
                for arg in arg_list:
                    arg_idx = arg[:2]
                    id_arg_span = get_span_idx(arg_idx, self.arg_span_idx)
                    label_id = self.labels_mapping[arg[-1]]
                    indice_pas = [idx_sent, id_pred_span, id_arg_span, label_id]
                    indices.append(indice_pas)
                    print(indice_pas)
                
        updates = [1 for i in indices]
        # self.output =     
                

    def extract_bert(self):
        self.bert_emb = extract_bert(self.bert_model, self.bert_tokenizer, self.sentences, self.max_tokens, self.padding_side)
        np.save('data/features/bert_emb.npy', self.bert_emb)

    def extract_emb_features(self):
        padded = pad_input(self.sentences, self.max_tokens)
        word_emb = np.ones(shape=(len(self.sentences), self.max_tokens, 300))
        for i, sent in enumerate(padded):
            for j, word in enumerate(sent):
                if (word == '<pad>'):
                    word_vec = np.zeros(300)
                else:
                    word_vec = self.word_vec[word.lower()]
                word_emb[i][j] = word_vec
        self.word_emb = word_emb
        np.save('data/features/word_emb.npy', word_emb)

    def read_raw_data(self):
        file = open(os.getcwd()+train_data_path + self.config['train_data'], 'r')
        lines = file.readlines()
        
        errorData = []
        for pairs in lines:
            # Split sent, label list
            sent, PASList = split_first(pairs, ';')
            tokens = sent.split(' ')
            # Current labels such as = A0, A1, O
            currentLabel = ''

            # start idx and end ix of each labels
            start_idx = -1
            end_idx = -1

            padding = self.max_tokens - len(tokens)

            arg_list = []

            # Looping each PAS for every predicate
            for PAS in PASList:
                # Array: (num_labels) : (B-AO, I-A0..)
                srl_labels = PAS.split(' ')
                # Check label length and sentence length
                if (len(srl_labels) != len(tokens)):
                    errorData.append(sent)
                    break

                sentence_args_pair = {
                    'id_pred': 0,
                    'pred': '',
                    'args': []
                }
                for idx, srl_label in enumerate(srl_labels):
                    # bio: B/I/O
                    bio, srl = split_first(srl_label, '-')

                    srl = ('-').join(srl) # (A0, A1, AM-TMP, ...)

                    if (bio == 'B' or bio == 'O'):
                        if (currentLabel != '' and currentLabel != srl):
                            end_idx = padding + idx - 1
                            if (currentLabel == 'V'):
                                sentence_args_pair['id_pred'] = [start_idx, end_idx]
                                sentence_args_pair['pred'] = sent.split(' ')[start_idx: end_idx+1]
                            else:
                                temp = [start_idx, end_idx, currentLabel]
                                sentence_args_pair['args'].append(temp)
                        start_idx = padding + idx
                        currentLabel = srl
                # Append for different PAS, same sentences
                arg_list.append(sentence_args_pair)

            # Append for differents sentences
            self.sentences.append(tokens)
            self.arg_list.append(arg_list)
        
