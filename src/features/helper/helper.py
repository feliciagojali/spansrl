import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split

core_arguments= ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4']
def extract_bert(model, tokenizer, sentences, max_tokens, device):
    bert_features = [bert_sent(sent, model, tokenizer, max_tokens, device) for  sent in sentences]
    return bert_features

def save_emb(emb, type, trainType, isSum=False):
    filename=''
    if (isSum):
        filename += 'sum_'
    filename += type
    np.save('data/features/' + filename + '.npy', emb)

def save_npy(filename, arr):
    np.save(filename, arr)

def bert_sent(sentence, model, tokenizer, max_tokens, device):
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
  
    x = torch.LongTensor(input_ids).view(1,-1).to(device)
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

def extract_pas_index(pas_list, tokens, max_tokens):
    # Looping each PAS for every predicate
    max_sent = -1
    arg_list = []
    for PAS in pas_list:
        PAS = PAS.strip()
        if (PAS == ''):
            continue
        # Array: (num_labels) : (B-AO, I-A0..)
        srl_labels_ = PAS.split(' ')
        # Check label length and sentence length
        if (len(srl_labels_) != len(tokens)):
            print('Tokens and labels do not sync = '+ str(tokens))
            continue

        srl_labels_, pad_sent = pad_input([srl_labels_, tokens], max_tokens, 'O')
        sentence_args_pair = {
            'id_pred': 0,
            'pred': '',
            'args': []
        }
        # Current labels such as = A0, A1, O
        srl_labels = [] 
        for srl in srl_labels_:
            if (srl == 'O'):
                srl_labels.append(['O','O'])
            else:
                bio, label = split_first(srl,'-')
                srl_labels.append([bio, '-'.join(label)])
        cur_label = 'O'
        # start idx and end ix of each labels
        start_idx = 0
        end_idx = 0

            
        for idx, srl_label in enumerate(srl_labels):
            if ((len(srl_label) == 2 and srl_label[0] != 'I') or len(srl_label) == 1):
                if (cur_label == 'REL'):
                    sentence_args_pair['pred'] = pad_sent.tolist()[start_idx: end_idx+1]
                    sentence_args_pair['id_pred'] = [start_idx,end_idx]
                elif (cur_label != 'O') :
                    temp = [start_idx, end_idx , cur_label]
                    sentence_args_pair['args'].append(temp)
                cur_label = srl_label[1] if len(srl_label) == 2 else 'O'
                start_idx = idx
            end_idx = idx
        # Handle last label
        if (cur_label == 'REL'):
            sentence_args_pair['pred'] = pad_sent.tolist()[start_idx: end_idx+1]
            sentence_args_pair['id_pred'] = [start_idx,end_idx]
        elif (cur_label != 'O') :
            temp = [start_idx, end_idx , cur_label]
            sentence_args_pair['args'].append(temp)
        # Append for different PAS, same sentences
        if(sentence_args_pair['id_pred'] == 0):
            print('These sentences do not have verb label in it ='+str(tokens))
            continue
        arg_list.append(sentence_args_pair)
    return arg_list, max_sent

def handle_overlapped_span(start, end, label, val, span):
    overlapped_keys = []
    for key in list(span.keys()):
        start_2, end_2, _ = list(key)
        arr = [x for x in range(max(start, start_2), min(end, end_2)+1)]
        # Overlap
        if len(arr) != 0:
            overlapped_keys.append(key)

    # Check if current val is greater than the value for all overlapped
    i = 0
    winning = True
    while (winning and i < len(overlapped_keys)):
        key = overlapped_keys[i]
        if (val < span[key]):
            winning = False
        i+=1
    # If greater than replace all overlapped
    if (winning):
        for key in overlapped_keys:
            del span[key]
        new_key = tuple([start, end, label])
        span[new_key] = val

    return winning, overlapped_keys, span

def convert_idx(ids, num_sent, arg_span_idx, pred_span_idx, labels_mapping, max_val, use_constraint, arg_idx_mask=None, pred_idx_mask=None):
    arr = [[] for _ in range(num_sent)]
    all_val = [[] for _ in range(num_sent)]
    cur_pred = -1
    cur_sent = -1
    for id in ids:
        sentence_args_pair = {
            'id_pred': 0,
            'args': []
        }
        sent, pred, arg, label = id
        current_arg = arg_idx_mask[sent][arg] if arg_idx_mask is not None else arg
        arg_id_start = arg_span_idx[0][current_arg]
        arg_id_end = arg_span_idx[1][current_arg]
        arg_span = [arg_id_start, arg_id_end, labels_mapping[label]]
        current_val = max_val[sent][pred][arg][0]
        if pred == cur_pred and sent == cur_sent :
            all_val[sent][-1].append(current_val)
            arr[sent][-1]['args'].append(arg_span)
        else :
            cur_pred = pred
            if (pred_idx_mask is not None):
                pred = pred_idx_mask[sent][pred]
            pred_id_start = pred_span_idx[0][pred]
            pred_id_end = pred_span_idx[1][pred]
            sentence_args_pair['id_pred'] = [pred_id_start, pred_id_end]
            sentence_args_pair['args'].append(arg_span)
            all_val[sent].append([current_val])
            arr[sent].append(sentence_args_pair)
        cur_sent = sent
    if (use_constraint):
        arr = filter_no_overlapping_spans(arr, all_val)
    return arr

def is_overlapped(range_1, range_2):
    s_1, e_1 = range_1
    s_2, e_2 = range_2
    
    arr = [x for x in range(max(s_1, s_2), min(e_1, e_2)+1)]
    
    if (len(arr)!=0):
        return True
    else:
        return False
    
def filter_no_overlapping_spans(arr, value):
    overlapped_pas = []
    for i, (sent, val_sent) in enumerate(zip(arr, value)):
        for j, (pas, val_pas) in enumerate(zip(sent, val_sent)):
            args = pas['args']
            overlapped = []
            
            for id_1, arg_1 in enumerate(args):
                for id_2 in range(id_1+1, len(args)):
                    arg_2 = args[id_2]
                    if (is_overlapped(arg_1[:2], arg_2[:2])):
                        if (len(overlapped) == 0):
                            overlapped.append([id_1, id_2])
                        else:
                            for k, o in enumerate(overlapped):
                                if id_1 in o or id_2 in o:
                                    overlapped[k].extend([id_1, id_2])
                                    break
            if (len(overlapped) != 0):
                for overlap in overlapped:
                    overlap = list(set(overlap))
                    val = [val_pas[x] for x in overlap]
                    sorted_overlap = [x for _, x in sorted(zip(val, overlap), key=lambda pair:pair[0], reverse=True)]
                    chosen = [sorted_overlap[0]]
                    not_chosen = sorted_overlap[1:]
                    for not_id in not_chosen:
                        id = 0
                        isOverlapped = False
                        while(id < len(chosen) and not isOverlapped):
                            current_id = chosen[id]
                            isOverlapped = is_overlapped(args[not_id][:2], args[current_id][:2])
                            id +=1
                        if (not isOverlapped):
                            chosen.append(not_id)
                    not_chosen = sorted(list(set(sorted_overlap) - set(chosen)))
                    for o in not_chosen:
                        overlapped_pas.append([i, j, o])
    overlapped_pas = sorted(overlapped_pas, key=lambda x:(-x[1], -x[2]))
    for id in overlapped_pas:
        i, j, o = id
        del arr[i][j]['args'][o]
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


    
def split_train_test_val(features_1, features_11, features_2, features_3, out, sentences, config):
    f1_train, f1_test,f11_train, f11_test, f2_train, f2_test, f3_train, f3_test, out_train, out_test, sent_train, sent_test= train_test_split(features_1, features_11, features_2, features_3, out, sentences, test_size=0.4,train_size=0.6)
    f1_test, f1_val, f11_test, f11_val, f2_test, f2_val, f3_test, f3_val, out_test,out_val, sent_test, sent_val= train_test_split(f1_test, f11_test, f2_test,f3_test,out_test,sent_test, test_size = 0.5,train_size =0.5)
    type = {
        "train": [f1_train, f11_train, f2_train, f3_train, out_train, sent_train],
        "test": [f1_test, f11_test, f2_test,f3_test, out_test, sent_test],
        "val": [f1_val, f11_val, f2_val, f3_val, out_val, sent_val]
    }
    dir = config['features_dir']
    filename = [config['features_1'], config['features_1.1'], config['features_2'],config['features_3'], config['output'], 'sentences.npy']
    for key, val in type.items():
        for typ, name in zip(val, filename):
            np.save(dir+str(key)+'_'+name, typ)

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


def pad_sentences(sentences, max_tokens, isArray=False):
    if (isArray):
        padded = [pad_input(sent, max_tokens) for sent in sentences]
    else:
        padded = pad_input(sentences, max_tokens)
    return padded