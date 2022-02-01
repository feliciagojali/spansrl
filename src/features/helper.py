from re import sub
from tqdm import tqdm
import numpy as np
import torch
import tensorflow as tf
## BERT functions
def extract_bert(model, tokenizer, sentences, max_tokens):
    bert_features = [bert_sent(sent, model, tokenizer, max_tokens) for  sent in sentences]
    return bert_features

def save_emb(emb, type, trainType, isSum=False):
    filename = trainType + '_'
    if (isSum):
        filename += 'sum_'
    filename += type
    np.save('data/features/' + filename + '.npy', emb)

def bert_sent(sentence, model, tokenizer, max_tokens):
    # Truncate
    if (len(sentence) > max_tokens):
        sentence = sentence[:max_tokens]

    # Get bert token (subword)
    tokens = tokenizer.tokenize(' '.join(sentence))
    
    # Get max length needed if word token
    max_len = max_tokens + len(tokens) - len(sentence) + 2       

    inputs = tokenizer(sentence, padding="max_length",max_length=max_len, is_split_into_words=True, truncation=True, return_offsets_mapping=True)
    
    # Remove bos, eos

    input_ids, offset = remove_sep(inputs, len(tokens))
  
    x = torch.LongTensor(input_ids).view(1,-1)
    out = model(x)[0].cpu().detach().numpy()
    
    # Handle subword (Average)
    # Get id of token which is subword
    is_subword =  [True if x[0] != 0 else False for x in offset]
    subword_list= [i for i, x in enumerate(is_subword) if x == True]
    
    if (len(subword_list) != 0):
        # total+=1
        start = subword_list[0]
        end = subword_list[0]
        # Id endpoints subword
        # arr = []
        sum = 0
        # Elements that're going to be deleted
        del_arr = []
        # New id to contain the average value
        new_id = []
        
        def add_data(new_id, del_arr, sum):
            if (len(del_arr) == 0):
                new_id.append(start-1)
            else :
                start_id = start - sum + len(new_id) -1
                new_id.append(start_id)
            # temp = [start-1, end]
            temp_del = [i for i in range(start-1, end+1)]
            # arr.append(temp)
            del_arr.append(temp_del)
            return new_id, del_arr, len(temp_del)

        for i, id in enumerate(subword_list):
            if (i != len(subword_list)-1):
                if (subword_list[i+1] == id + 1):
                    end = id + 1
                else:
                    new_id, del_arr,temp = add_data(new_id, del_arr, sum)
                    sum += temp
                    end = subword_list[i+1]
                    start = subword_list[i+1]
            else:
                if (id == subword_list[i-1] + 1):
                    end = id
                new_id, del_arr,temp = add_data(new_id, del_arr, sum)
                sum += temp
                end = id
                start = id
        # start_time = time.time()
        el_del = [item for sublist in del_arr for item in sublist[1:]]
        mean_value = [np.mean(out[0][x[0]:x[-1]], axis=0) for x in del_arr]
        # Prepare out vector
        filtered_out = np.delete(out, el_del, axis=1)
        # Insert value
        for id, vec in zip(new_id, mean_value):
            filtered_out[0][id] = vec
        # after += time.time() - start_time
    else:
        filtered_out = out
    return filtered_out[0]

def remove_sep(inputs, len_tokens):
    ids = inputs['input_ids']
    offset_ids = inputs['offset_mapping']
    start_id = 0
    end_id = start_id + len_tokens + 1
    del ids[end_id]
    del ids[start_id]
    del offset_ids[end_id]
    del offset_ids[start_id]
    return ids, offset_ids

def extract_pas_index(pas_list, tokens, max_tokens, max):
    # Looping each PAS for every predicate
    max_sent = -1
    arg_list = []
    for PAS in pas_list:
        PAS = PAS.strip()
        # Array: (num_labels) : (AO, A0..)
        srl_labels = PAS.split(' ')
        # Check label length and sentence length
        if (len(srl_labels) != len(tokens)):
            print('Tokens and labels do not sync = '+ str(tokens))
            continue

        srl_labels, pad_sent = pad_input([srl_labels, tokens], max_tokens, 'O')
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
                if (cur_label == 'REL'):
                    sentence_args_pair['pred'] = pad_sent.tolist()[start_idx: end_idx+1]
                    sentence_args_pair['id_pred'] = [start_idx,end_idx]
                elif (cur_label != 'O') :
                    temp = [start_idx, end_idx , cur_label]
                    if (end_idx-start_idx + 1 > max):
                        max = end_idx-start_idx + 1
                        max_sent = tokens
                    sentence_args_pair['args'].append(temp)
                cur_label = srl_label
                start_idx = idx
            end_idx = idx
        # Handle last label
        if (cur_label == 'REL'):
            sentence_args_pair['pred'] = pad_sent.tolist()[start_idx: end_idx+1]
            sentence_args_pair['id_pred'] = [start_idx,end_idx]
        elif (cur_label != 'O') :
            if (end_idx-start_idx + 1 > max):
                max = end_idx-start_idx + 1
                max_sent = tokens
            temp = [start_idx, end_idx , cur_label]
            sentence_args_pair['args'].append(temp)
        # Append for different PAS, same sentences
        if(sentence_args_pair['id_pred'] == 0):
            print('These sentences do not have verb label in it ='+str(tokens))
            continue
        arg_list.append(sentence_args_pair)
    return arg_list, max, max_sent

def convert_idx(ids, num_sent, arg_span_idx, pred_span_idx, labels_mapping, arg_idx_mask=None, pred_idx_mask=None):
    arr = [[] for _ in range(num_sent)]
    cur_pred = 0
    for id in ids:
        sentence_args_pair = {
            'id_pred': 0,
            'args': []
        }
        sent, pred, arg, label = id
        if (arg_idx_mask):
            arg = arg_idx_mask[sent][arg]
        arg_id_start = arg_span_idx[0][arg]
        arg_id_end = arg_span_idx[1][arg]
        arg_span = [arg_id_start, arg_id_end, labels_mapping[label]]
        if pred == cur_pred :
            arr[sent][-1]['args'].append(arg_span)
        else :
            if (pred_idx_mask):
                pred = pred_idx_mask[sent][pred]
            pred_id_start = pred_span_idx[0][pred]
            pred_id_end = pred_span_idx[1][pred]
            sentence_args_pair['id_pred'] = [pred_id_start, pred_id_end]
            sentence_args_pair['args'].append(arg_span)
            arr[sent].append(sentence_args_pair)
        cur_pred = pred
    return arr

def pad_input(data, max_token, pad_char='<pad>'):
    padded_data = np.full(shape=(len(data), max_token), fill_value=pad_char, dtype='object')
    for i, sent in enumerate(data):
        if (len(sent) >= max_token):
            padded_data[i] = sent[:max_token]
            continue
        padded_data[i][:len(sent)] = sent
    return padded_data

def create_dict(data, switch=False):
    dict = {}
    for line in tqdm(data):
        if (switch):
            val, key = line.split('	')
        else :
            key, val = line.split('	')
        dict[int(key)] = val
    return dict

def split_first(data, separator):
    arr = data.split(separator)
    first = arr[0]
    rest = arr[1:]
    return first, rest

def label_encode(label_list, init=False):
    if (init):
        label_dict = init
    else:
        label_dict = {}

    for i in label_list:
        label_dict[i] = len(label_dict)
    
    return label_dict

def get_span_idx(input_idx, span_idx):
    start, end = input_idx
    start_idx, end_idx, _ = span_idx

    idx = -1
    # idx = [Input2.index(y) for x, y in zip(Input1, Input2) if x == 'ss' and y == 'fo']
    for i, (s, e) in enumerate(zip(start_idx, end_idx)):
        if (start == s and end == e):
            idx = i
            break
    
    return idx

def check_pred_id(pred_id, pas_list):
    arg_list = [x['args'] for x in pas_list if x['id_pred'] == pred_id]
    return arg_list

def _print_f1(total_gold, total_predicted, total_matched, message=""):
    precision = 100.0 * total_matched / total_predicted if total_predicted > 0 else 0
    recall = 100.0 * total_matched / total_gold if total_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    print ("{}: Precision: {}, Recall: {}, F1: {}".format(message, precision, recall, f1))
    return precision, recall, f1