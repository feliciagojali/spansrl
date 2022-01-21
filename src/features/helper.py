from tqdm import tqdm
import numpy as np
import torch
## BERT functions
def extract_bert(model, tokenizer, sentences, max_tokens, pad_side):
    bert_features = []
    for sentence in sentences:
        # Get bert token (subword)
        tokens = tokenizer(' '.join(sentence))
        print(tokens)
        # Get max length needed if word token
        max_len = max_tokens + len(tokens) - len(sentence) + 2
        print(max_len)
        # Total padding
        num_pad = max_len - len(tokens) - 2
        ## TO DO: Handle Truncating
        inputs = tokenizer(sentence, padding="max_length",max_length=max_len, is_split_into_words=True, truncation=True, return_offsets_mapping=True)
        # Remove bos, eos
        input_ids, offset = remove_sep(inputs, num_pad, len(tokens), pad_side)
        print(len(inputs['input_ids']))
        x = torch.LongTensor(input_ids).view(1,-1)
        out = model(x)[0].cpu().detach().numpy()

        # Handle subword (Average)
        # Get id of token which is subwor
        is_subword = np.array(offset)[:,0] != 0
        subword_list = np.where(is_subword == True)
        subword_list = subword_list[0].tolist()

        if (len(subword_list) != 0):
            start = subword_list[0]
            end = subword_list[0]
            # Id endpoints subword
            arr = []
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
                temp = [start-1, end]
                temp_del = [i for i in range(start-1, end+1)]
                arr.append(temp)
                del_arr.append(temp_del)
                return new_id, arr, del_arr, len(temp_del)

            for i, id in enumerate(subword_list):
                if (id == end + 1):
                    end = id
                    if (i == len(subword_list) -1):
                        new_id, arr, del_arr,temp = add_data(new_id, arr, del_arr, sum)
                        sum += temp
                else :
                    if (i - 1 >= 0):
                        new_id, arr, del_arr,temp = add_data(new_id, arr, del_arr, sum)
                        sum += temp
                        end = id
                        start = id

            el_del = [item for sublist in del_arr for item in sublist[1:]]
            mean_value = [np.mean(out[0][i:j+1], axis=0) for i,j in arr]
            # Prepare out vector
            filtered_out = np.delete(out, el_del, axis=1)
            # Insert value
            for id, vec in zip(new_id, mean_value):
                filtered_out[0][id] = vec
            else:
                filtered_out = out
            out = filtered_out[0]
        bert_features.append(out)
        print(out.shape)
    return bert_features

        
def remove_sep(inputs, pad, len_tokens, pad_type='left'):
    ids = inputs['input_ids']
    offset_ids = inputs['offset_mapping']
    if (pad_type == 'left'):
        start_id = pad 
    else:
        start_id = 0
    end_id = start_id + len_tokens + 1
    sep = [start_id, end_id]
    new_ids = np.delete(ids,sep)
    new_offset = np.delete(offset_ids, sep, axis=0)
    return new_ids, new_offset

def pad_input(data, max_token, pad_char='<pad>'):
    padded_data = np.full(shape=(len(data), max_token), fill_value=pad_char, dtype='object')
    for i, sent in enumerate(data):
        idx_start = max_token - len(sent)
        padded_data[i][idx_start:] = sent
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

def label_encode(label_list):
    label_dict = {}

    for i in label_list:
        label_dict[i] = len(label_dict)
    
    return label_dict

def get_span_idx(input_idx, span_idx):
    start, end = input_idx
    start_idx, end_idx, _ = span_idx

    idx = -1
    for i, (s, e) in enumerate(zip(start_idx, end_idx)):
        if (start == s and end == e):
            idx = i
            break
    
    return idx
