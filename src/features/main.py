raw_path = 'data/raw/'
processed_path = 'data/processed/'
raw_filename = 'raw.csv'

import tensorflow as tf
import json
import numpy as np
from SRLData import SRLData
import pandas as pd

def main():
    filename = './configurations.json'
    path = 'data/raw/'
    types = ['val']
    f = open(filename)
    all_config = json.load(f)

    data = SRLData(all_config['default'], False)

    # files = [path+'dev_summary_corpus.csv']

    # datas = [pd.read_csv(file)['article'] for file in files]
    # for i in datas:
    #     data.extract_features(i, 'val', True)
   
    data.read_raw_data()
    # data.extract_features("train")
    # data.convert_train_output()
        
   


if __name__ == "__main__":
    main()