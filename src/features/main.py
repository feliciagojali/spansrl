raw_path = 'data/raw/'
processed_path = 'data/processed/'
raw_filename = 'raw.csv'

import enum
import json

from SRLData import SRLData

def main():
    filename = './configurations.json'
    f = open(filename)
    all_config = json.load(f)

    data = SRLData(all_config['default'])
    data.sentences = [["saya","dimaafkan","saya","dimaafkan","saya","dimaafkan","saya","dimaafkan","saya","dimaafkan","saya","dimaafkan"],["tahu","malu","dimaafkan"]]
    data.extract_features()


if __name__ == "__main__":
    main()