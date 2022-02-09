raw_path = 'data/raw/'
processed_path = 'data/processed/'
raw_filename = 'raw.csv'

# import tensorflow as tf
import json
import numpy as np
from helper import SRLData, split_into_batch
import ast
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
def main():
    filename = './configurations.json'
    # path = 'data/raw/'
    # types = ['val']
    f = open(filename)
    all_config = json.load(f)
    types = 'train'
    # raw_data = '../data/raw/'+types+'_summary_corpus.csv'
    # data = pd.read_csv(raw_data)
    # data['article'] = data['article'].progress_apply(lambda x : ast.literal_eval(x))

    # split_into_batch(data['article'], 6, types)
    data = SRLData(all_config['summary'])
    path = '../data/raw/'+types+'/'
    i = [x+1 for x in range(90)]
    for id in i:
        file = path+types+'_sum_sent_' + str(id)+'.npy'

        datas = np.load(file, allow_pickle=True)

        # datas['article'] = datas['article'].progress_apply(lambda x : ast.literal_eval(x))
        # for i in datas:
        #     data.extract_features(i, 'val', True)
    
        # data.read_raw_data()
        data.extract_features(datas, types, id, True)
        # data.convert_train_output()
            
   


if __name__ == "__main__":
    main()