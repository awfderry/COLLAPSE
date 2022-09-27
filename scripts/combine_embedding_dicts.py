import pickle
import glob
import sys
import numpy as np
import collections as col

in_files = glob.glob(sys.argv[1])
outfile = sys.argv[2]

with open('../data/af2_humanembeddings_14.pkl', 'rb') as f:
    d = pickle.load(f)

newdict = col.defaultdict(list)
for dictfile in in_files:
    print(dictfile)
    with open(dictfile, 'rb') as f:
        d = pickle.load(f)
    for k,v in d.items():
        if isinstance(v, list):
            newdict[k].extend(list(v))
        else:
            newdict[k].append(v)

newdict['embeddings'] = np.concatenate(newdict['embeddings'])
print('combined embedding size:', newdict['embeddings'].shape)
with open(outfile, 'wb') as f:
    pickle.dump(newdict, f, protocol=4)