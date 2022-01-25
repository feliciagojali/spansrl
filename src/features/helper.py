from re import sub
from tqdm import tqdm
import numpy as np
import torch
import time
for_loop = 0
after = 0
total = 0
## BERT functions
def extract_bert(model, tokenizer, sentences, max_tokens, pad_side):
    bert_features = [bert_sent(sent, model, tokenizer, max_tokens, pad_side) for  sent in sentences]
    return bert_features

def bert_sent(sentence, model, tokenizer, max_tokens, pad_side):
    # Truncate
    if (len(sentence) > max_tokens):
        sentence = sentence[:max_tokens]

    # Get bert token (subword)
    tokens = tokenizer.tokenize(' '.join(sentence))
    
    # Get max length needed if word token
    max_len = max_tokens + len(tokens) - len(sentence) + 2       

    # Total padding
    num_pad = max_len - len(tokens) - 2
    inputs = tokenizer(sentence, padding="max_length",max_length=max_len, is_split_into_words=True, truncation=True, return_offsets_mapping=True)
    
    # Remove bos, eos

    input_ids, offset = remove_sep(inputs, num_pad, len(tokens), pad_side)
  
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
        
        def add_data(new_id, arr, del_arr, sum):
            if (len(del_arr) == 0):
                new_id.append(start-1)
            else :
                start_id = start - sum + len(new_id) -1
                new_id.append(start_id)
            # temp = [start-1, end]
            temp_del = [i for i in range(start-1, end+1)]
            # arr.append(temp)
            del_arr.append(temp_del)
            return new_id, arr, del_arr, len(temp_del)

        for i, id in enumerate(subword_list):
            if (i != len(subword_list)-1):
                if (subword_list[i+1] == id + 1):
                    end = id + 1
                else:
                    new_id, arr, del_arr,temp = add_data(new_id, arr, del_arr, sum)
                    sum += temp
                    end = subword_list[i+1]
                    start = subword_list[i+1]
            else:
                if (id == subword_list[i-1] + 1):
                    end = id
                new_id, arr, del_arr,temp = add_data(new_id, arr, del_arr, sum)
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

def remove_sep(inputs, pad, len_tokens, pad_type='left'):
    ids = inputs['input_ids']
    offset_ids = inputs['offset_mapping']
    if (pad_type == 'left'):
        start_id = pad 
    else:
        start_id = 0
    end_id = start_id + len_tokens + 1
    del ids[end_id]
    del ids[start_id]
    del offset_ids[end_id]
    del offset_ids[start_id]
    return ids, offset_ids

def remove_new(inputs):
    ids = inputs['input_ids']
    offset_ids = inputs['offset_mapping']
    start_id = ids.index(2)
    ids.pop(start_id)
    offset_ids.pop(start_id)
    end_id = ids.index(3)
    ids.pop(end_id)
    offset_ids.pop(end_id)  
    return ids, offset_ids

def pad_input(data, max_token, pad_char='<pad>', pad_type='left'):
    padded_data = np.full(shape=(len(data), max_token), fill_value=pad_char, dtype='object')
    for i, sent in enumerate(data):
        if (len(sent) >= max_token):
            padded_data[i] = sent[:max_token]
            continue
    
        if (pad_type == 'left'):
            idx_start = max_token - len(sent)
        else:
            idx_start = 0

        padded_data[i][idx_start:idx_start+len(sent)+1] = sent
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
