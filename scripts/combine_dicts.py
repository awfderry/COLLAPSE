import pickle
import glob
import sys

in_files = glob.glob(sys.argv[1])
outfile = sys.argv[2]
combined_dict = {}

for dictfile in in_files:
    print(dictfile)
    with open(dictfile, 'rb') as f:
        d = pickle.load(f)
    
    combined_dict.update(d)

with open(outfile, 'wb') as f:
    pickle.dump(combined_dict, f)