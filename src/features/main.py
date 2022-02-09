raw_path = 'data/raw/'
processed_path = 'data/processed/'
raw_filename = 'raw.csv'

# import tensorflow as tf
from curses import raw
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
    # all_config = json.load(f)
    raw_data = '../data/raw/dev_summary_corpus.csv'
    data = pd.read_csv(raw_data)
    data['article'] = data['article'].progress_apply(lambda x : ast.literal_eval(x))

    split_into_batch(data['article'], 6)
    # data = SRLData(all_config['summary'])
    # path = 'data/processed/train/batched/'
    # i = [1, 2, 3, 4, 5, 6]
    # for id in i:
    #     for k in range(2):
    #         file = path+'train_sum_sent_' + str(id)+'.'+str(k+1) +'.npy'

    #         datas = np.load(file, allow_pickle=True)

    #         # datas['article'] = datas['article'].progress_apply(lambda x : ast.literal_eval(x))
    #         # for i in datas:
    #         #     data.extract_features(i, 'val', True)
        
    #         # data.read_raw_data()
    #         data.extract_features(datas, "test", id,k,  True)
            # data.convert_train_output()
            
   


if __name__ == "__main__":
    main()