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
    # data.read_raw_data()
    # data.extract_features("train")
    x = np.load(all_config['default']['train_out'])
    out = data.convert_result_to_readable(x[:10])
    for i in out:
        print(i)
        
   


if __name__ == "__main__":
    main()