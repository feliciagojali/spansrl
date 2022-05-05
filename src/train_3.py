import numpy as np
from regex import P

types = ['train', 'val', 'test']

for t in types:
    print(t)
    data = np.load('data/features/'+t +'_full_sentences.npy')
    print(len(data))
    with open('data/results/'+ t +'_' + 'sentences.txt', 'w') as f:
        for item in data:
            f.write("%s" %str(item))
