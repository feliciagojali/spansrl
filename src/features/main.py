raw_path = 'data/raw/'
processed_path = 'data/processed/'
raw_filename = 'raw.csv'

import tensorflow as tf
import json
import numpy as np
from SRLData import SRLData

def main():
    filename = './configurations.json'
    f = open(filename)
    all_config = json.load(f)

    data = SRLData(all_config['default'])
    data.read_raw_data()
    # data.extract_features("train")
    data.convert_train_output()
   


if __name__ == "__main__":
    main()