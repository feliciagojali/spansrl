

import os
import sys

from pkg_resources import ExtractionError


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
        ## Character
        initial_char_dict = {"<pad>":0, "<unk>":1}
        self.char_dict = label_encode(config['char_list'], initial_char_dict)
        self.char_input = []
        ## Word Embedding
        self.padded_sent = []
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
        pred_start, _, _ = self.pred_span_idx
        arg_start, _, _ = self.arg_span_idx

        num_preds = len(pred_start)
        num_args = len(arg_start)
        # Fill with null labels first
        # initialTensor = tf.fill([batch_size, num_preds, num_args, self.num_labels+1], len(self.labels_mapping) + 1)
        initialData = np.zeros([batch_size, num_preds, num_args, self.num_labels])
        initialLabel = np.ones([batch_size, num_preds, num_args, 1])
        initialData = np.concatenate([initialData, initialLabel], axis=-1)
        indices = []
        indices_null = []
        for idx_sent, sentences in enumerate(self.arg_list):
            for PAS in sentences:
                id_pred_span = get_span_idx(PAS['id_pred'], self.pred_span_idx)
                if (id_pred_span == -1):
                    continue
                arg_list = PAS['args']
                for arg in arg_list:
                    arg_idx = arg[:2]
                    id_arg_span = get_span_idx(arg_idx, self.arg_span_idx)
                    if (id_arg_span == -1):
                        continue
                    label_id = self.labels_mapping[arg[-1]]
                    indice_pas = [idx_sent, id_pred_span, id_arg_span, label_id]
                    indice_reset = [idx_sent, id_pred_span, id_arg_span, self.num_labels + 1]
                    indices.append(indice_pas)
                    indices_null.append(indice_reset)
                    print(indice_pas)
        print(indices)
        print(indices_null)       
        updates = [1 for _ in indices]
        updates_null = [0 for _ in indices_null]
        # initialData =  tf.tensor_scatter_nd_update(initialData, indices_null, updates_null)
        # self.output = tf.tensor_scatter_nd_update(initialData, indices, updates)
    

    def extract_features(self, isTraining=True, isSum=False):
        self.pad_sentences(isArray=isSum and not isTraining)
        if (isSum):
            # berishin dulu
            cleaned_sent = []
            if (isTraining):
                sentences = np.array(cleaned_sent).flatten()
                self.bert_emb = self.extract_bert_emb(sentences)
                self.word_emb = self.extract_emb(sentences, self.padded_sent)
                self.char_input = self.extract_char(sentences)
            else:
                # Documents
                self.bert_emb = [self.extract_bert_emb(sent) for sent in cleaned_sent]
                self.word_emb = [self.extract_emb(sent, padded) for sent, padded in zip(cleaned_sent, self.padded_sent)]   
                self.char_input = [self.extract_char(sent) for sent in cleaned_sent]
        else:
            self.bert_emb = self.extract_bert_emb(self.sentences)
            self.word_emb = self.extract_emb(self.sentences, self.padded_sent)
            self.char_input = self.extract_char(self.padded_sent)

        self.save_emb(self.bert_emb, 'bert_emb', isTraining, isSum)
        self.save_emb(self.word_emb, 'word_emb', isTraining, isSum)
        self.save_emb(self.char_input, 'char_input', isTraining, isSum)
        
    def extract_bert_emb(self, sentences): # sentences : Array (sent)
        bert_emb = extract_bert(self.bert_model, self.bert_tokenizer, sentences, self.max_tokens, self.padding_side)
        bert_emb = np.array(bert_emb)
        print(bert_emb)
        print(bert_emb.shape)
        return bert_emb
        

    def extract_emb(self, sentences, padded_sent):  # sentences : Array (sent)
        word_emb = np.ones(shape=(len(sentences), self.max_tokens, 300))
        for i, sent in enumerate(padded_sent):
            for j, word in enumerate(sent):
                if (word == '<pad>'):
                    word_vec = np.zeros(300)
                else:
                    word_vec = self.word_vec[word.lower()]
                word_emb[i][j] = word_vec
        print(word_emb.shape)
        print(word_emb)
        return word_emb
    
    def extract_char(self, sentences): # sentences: Array (sent)
        char = np.zeros(shape=(len(sentences), self.max_tokens, self.max_char))
        for i, sent in enumerate(sentences):
            for j, word in enumerate(sent):
                if (word == '<pad>'):
                    continue
                char_encoded = [self.char_dict[x]  if x in self.char_dict else self.char_dict['<unk>'] for x in word]
                if (len(char_encoded) >= self.max_char):
                    char[i][j] = char_encoded[:self.max_char]
                else:
                    char[i][j][:len(char_encoded)] = char_encoded
        print(char.shape)
        print(char)
        return char

    def pad_sentences(self, isArray=False):
        if (isArray):
            padded = [pad_input(sent, self.max_tokens, pad_type=self.padding_side) for sent in self.sentences]
        else:
            padded = pad_input(self.sentences, self.max_tokens, pad_type=self.padding_side)
        self.padded_sent = padded

    def save_emb(self, emb, type, isTraining=True, isSum=False):
        if (isTraining):
            filename = 'train_'
        else:
            filename = 'test_'
        if (isSum):
            filename += 'sum_'
        filename += type
        np.save('data/features/' + filename + '.npy', emb)

    def read_raw_data(self):
        file = open(os.getcwd() + self.config['train_data'], 'r')
        lines = file.readlines()
        
        errorData = []
        for pairs in lines:
            # Split sent, label list
            sent, PASList = split_first(pairs, ';')
            tokens = sent.split(' ')

            arg_list = []

            # Looping each PAS for every predicate
            for PAS in PASList:
                # Array: (num_labels) : (AO, A0..)
                srl_labels = PAS.split(' ')

                # Check label length and sentence length
                if (len(srl_labels) != len(tokens)):
                    errorData.append(sent)
                    break

                srl_labels, pad_sent = pad_input([srl_labels, tokens], self.max_tokens, 'O', self.padding_side)
                sentence_args_pair = {
                    'id_pred': 0,
                    'pred': '',
                    'args': []
                }
                # Current labels such as = A0, A1, O
                cur_label = srl_labels[0]
                # start idx and end ix of each labels
                start_idx = 0
                end_idx = 0
                for idx, srl_label in enumerate(srl_labels):
                    if (srl_label != cur_label):
                        if (cur_label == 'V'):
                            sentence_args_pair['pred'] = pad_sent.tolist()[start_idx: end_idx+1]
                            sentence_args_pair['id_pred'] = [start_idx,end_idx]
                        elif (cur_label != 'O') :
                            temp = [start_idx, end_idx , cur_label]
                            sentence_args_pair['args'].append(temp)
                        cur_label = srl_label
                        start_idx = idx
                    end_idx = idx
                # Handle last label
                if (cur_label == 'V'):
                    sentence_args_pair['pred'] = pad_sent.tolist()[start_idx: end_idx+1]
                    sentence_args_pair['id_pred'] = [start_idx,end_idx]
                elif (cur_label != 'O') :
                    temp = [start_idx, end_idx , cur_label]
                    sentence_args_pair['args'].append(temp)
                # Append for different PAS, same sentences
                arg_list.append(sentence_args_pair)

            # Append for differents sentences
            self.sentences.append(tokens)
            self.arg_list.append(arg_list)
        print(self.sentences)
        print(self.arg_list)
        
