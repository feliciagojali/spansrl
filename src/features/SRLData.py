

import os
import sys
from collections import Counter

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from features.helper import save_npy, _print_f1, check_pred_id, split_first, label_encode, get_span_idx, pad_input, extract_bert, extract_pas_index, save_emb, convert_idx
from utils.utils import create_span
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from gensim.models import fasttext, Word2Vec
import time
import sys
import torch

class SRLData(object):
    def __init__(self, config, emb=True):
        self.config = config
        # Configurations and constants
        self.max_arg_span = config['max_arg_span']
        self.max_pred_span = config['max_pred_span']
        self.max_tokens = config['max_tokens']
        self.num_labels = len(config['srl_labels'])
        self.max_char = config['max_char']
        self.arg_span_idx = create_span(self.max_tokens, self.max_arg_span)
        self.pred_span_idx = create_span(self.max_tokens, self.max_pred_span)

        # Features Input
        ## Character
        initial_char_dict = {"<pad>":0, "<unk>":1}
        self.char_dict = label_encode(config['char_list'], initial_char_dict)
        self.char_input = []

        ## Word Embedding
        self.use_fasttext = config['use_fasttext']
        self.emb1_dim = 300
        if (emb):
            # self.fast_text = fasttext.load_facebook_vectors(config['fasttext_emb_path'])
            # self.word_emb_ft = []
            # self.word_vec = Word2Vec.load(config['word_emb_path']).wv
            # self.word_emb_w2v = []

            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.bert_model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1").to(device)
            self.bert_tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1", padding_side='right')
            self.word_emb_2 = []
            self.emb2_dim = 768
        # Output
        self.output = []

        # Dict
        self.labels_mapping = label_encode(config['srl_labels'])


    # To convert train data labels to output models
    def convert_train_output(self):
        sentences = np.load(self.config['processed_sent'], allow_pickle=True)
        arg_list = np.load(self.config['processed_arg_list'], allow_pickle=True)
        batch_size = len(sentences)
        pred_start, _, _ = self.pred_span_idx
        arg_start, _, _ = self.arg_span_idx

        num_preds = len(pred_start)
        num_args = len(arg_start)
        # Fill with null labels first
        initialData = np.zeros([batch_size, num_preds, num_args, self.num_labels], dtype='int16')
        initialLabel = np.ones([batch_size, num_preds, num_args, 1], dtype='int16')
        initialData = np.concatenate([initialData, initialLabel], axis=-1)
        indices = []
        for idx_sent, sentences in (enumerate(tqdm(arg_list))):
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
                    try :
                        label_id = self.labels_mapping[arg[-1]]
                    except:
                        print(arg[-1])
                        continue
                    indice_pas = [idx_sent, id_pred_span, id_arg_span, label_id]
                    # max length = num_labels + 1
                    indices.append(indice_pas)
        for id in indices:
            sent, num_pred, num_spans, idx = id
            initialData[sent][num_pred][num_spans][idx] = 1
            initialData[sent][num_pred][num_spans][self.num_labels] = 0
        save_emb(initialData, "train", "output")
        print(initialData.shape)

    def extract_features(self, sentences, type, id, isSum=False):
        # sentences = np.load(self.config['processed_sent'], allow_pickle=True)
        # self.pad_sentences(sentences, isArray=isSum)
        if (isSum):
            # padded_sent = np.load(self.config['processed_padded_sent'], allow_pickle=True)
            # self.word_emb_ft = [self.extract_ft_emb(padded) for padded in (padded_sent)]   
            # self.word_emb_w2v = [self.extract_word_emb(padded) for padded in (padded_sent)]
            # self.char_input = [self.extract_char(sent) for sent in padded_sent]
            self.word_emb_2 = [self.extract_bert_emb(sent) for sent in tqdm(sentences, position=0, leave=True)]  
        else:
            padded_sent = np.load(self.config['processed_padded_sent'], allow_pickle=True)
            self.word_emb_w2v = self.extract_word_emb(padded_sent)
            self.word_emb_ft = self.extract_ft_emb(padded_sent)
            # self.word_emb_2 = self.extract_bert_emb(sentences)    
            # self.char_input = self.extract_char(padded_sent)

    
        # save_emb(self.word_emb_w2v, 'word_emb_w2v_1', type, isSum)
        # save_emb(self.word_emb_ft, 'word_emb_ft_15', type, isSum)
        np.save(type+ "_sum_bert_"+ str(id) +".npy", self.word_emb_2)
        # save_emb(self.word_emb_2, 'bert', type, isSum)
        # save_emb(self.char_input, 'char_input_5', type, isSum)
        # print(self.word_emb.shape)
        # print(self.word_emb_2.shape)
        # print(self.char_input.shape)
    
    def extract_word_emb(self, padded_sent):
        word_emb = np.zeros(shape=(len(padded_sent), self.max_tokens, 300), dtype='float32')
        for i, sent in enumerate(padded_sent):
            for j, word in enumerate(sent):
                if (word == '<pad>' or not self.word_vec.has_index_for(word.lower())):
                    continue
                word_vec = self.word_vec[word.lower()]
                word_emb[i][j] = word_vec
        return word_emb        

    def extract_sec_emb(self, sentences):
        if (self.use_fasttext):
            return self.extract_ft_emb(sentences, self.padded_sent)
        else:
            return self.extract_bert_emb(sentences)
    def extract_bert_emb(self, sentences): # sentences : Array (sent)
        bert_emb = extract_bert(self.bert_model, self.bert_tokenizer, sentences, self.max_tokens, self.device)
        bert_emb = np.array(bert_emb)
        return bert_emb
        

    def extract_ft_emb(self, padded_sent):  # sentences : Array (sent)
        word_emb = np.ones(shape=(len(padded_sent), self.max_tokens, 300), dtype='float32')
        for i, sent in (enumerate(padded_sent)):
            for j, word in enumerate(sent):
                if (word == '<pad>'):
                    word_vec = np.zeros(300,dtype='int8')
                else:
                    word_vec = self.fast_text[word.lower()]
                word_emb[i][j] = word_vec
        return word_emb
    
    def extract_char(self, sentences): # sentences: Array (sent)
        char = np.zeros(shape=(len(sentences), self.max_tokens, self.max_char), dtype='int8')
        for i, sent in tqdm(enumerate(sentences)):
            for j, word in enumerate(sent):
                if (word == '<pad>'):
                    continue
                char_encoded = [self.char_dict[x]  if x in self.char_dict else self.char_dict['<unk>'] for x in word]
                if (len(char_encoded) >= self.max_char):
                    char[i][j] = char_encoded[:self.max_char]
                else:
                    char[i][j][:len(char_encoded)] = char_encoded
        return char

    def pad_sentences(self, sentences, isArray=False):
        if (isArray):
            padded = [pad_input(sent, self.max_tokens) for sent in sentences]
        else:
            padded = pad_input(sentences, self.max_tokens)
        save_npy(self.config['processed_padded_sent'], padded)

    def convert_result_to_readable(self, out, arg_mask=None, pred_mask=None): # (batch_size, num_preds, num_args, num_labels)
        labels_list = list(self.labels_mapping.keys())
        ## Max = 1
        max_val = tf.reduce_max(out, axis=-1, keepdims=True)
        cond = tf.equal(out, max_val)
        out_ = tf.where(cond, tf.ones_like(out), tf.zeros_like(out))
        ## Get Id
        transpose = tf.transpose(out_, [3, 0, 1, 2])
        omit = tf.transpose(transpose[:-1], [1, 2, 3, 0])
        #array of position
        ids = tf.where(omit)
        pas = convert_idx(ids, len(out), self.arg_span_idx, self.pred_span_idx, labels_list, arg_mask, pred_mask)
        return pas

    def evaluate(self, y, pred):
        # Adopted from unisrl
        total_gold = 0
        total_pred = 0
        total_matched = 0
        total_unlabeled_matched = 0
        comp_sents = 0
        label_confusions = Counter()
        total_gold_label = Counter()
        total_pred_label = Counter()
        total_matched_label = Counter()
        for y_sent, pred_sent in zip(y, pred):
            gold_rels = 0
            pred_rels = 0
            matched = 0
            for gold_pas in y_sent:
                pred_id = gold_pas['id_pred']
                gold_args = gold_pas['args']
                total_gold += len(gold_args)
                gold_rels += len(gold_args)
                total_gold_label.update(['V'])
                
                arg_list_in_predict = check_pred_id(pred_id, pred_sent)
                if (len(arg_list_in_predict) == 0):
                    continue
                total_matched_label.update(['V'])
                for arg0 in gold_args:
                    label = arg0[-1]
                    total_gold_label.update([label])
                    for arg1 in arg_list_in_predict[0]:
                        if (arg0[:-1] == arg1[:-1]): # Right span
                            total_unlabeled_matched += 1
                            label_confusions.update([(arg0[2], arg1[2]),])
                            if (arg0[2] == arg1[2]): # Right label
                                total_matched_label.update([arg0[2]])
                                total_matched += 1
                                matched += 1
            for pred_pas in pred_sent:
                total_pred_label.update(['V'])
                pred_id = pred_pas['id_pred']
                pred_args = pred_pas['args']
                for arg1 in pred_args:
                    label = arg1[-1]
                    total_pred_label.update([label])
                total_pred += len(pred_args)
                pred_rels += len(pred_args)
            
            if (gold_rels == matched and pred_rels == matched):
                comp_sents += 1

        precision, recall, f1 = _print_f1(total_gold, total_pred, total_matched, "SRL")
        ul_prec, ul_recall, ul_f1 = _print_f1(total_gold, total_pred, total_unlabeled_matched, "Unlabeled SRL")
        total = total_gold_label + total_pred_label
        for key in total:
            _print_f1(total_gold_label[key], total_pred_label[key], total_matched_label[key], str(key))

        return


    def read_raw_data(self):
        file = open(os.getcwd() +self.config['train_data'], 'r')
        lines = file.readlines()
        max = 0
        sentences = []
        arg_lists = []
        for pairs in lines:
            # Split sent, label list
            sent, PASList = split_first(pairs, ';')
            tokens = sent.split(' ')
            arg_list, max, sent = extract_pas_index(PASList, tokens, self.max_tokens, max)

            # Append for differents sentences
            sentences.append(tokens)
            arg_lists.append(arg_list)
        save_npy(self.config['processed_sent'], sentences)
        save_npy(self.config['processed_arg_list'], arg_lists)

        
