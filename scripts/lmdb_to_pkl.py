import argparse
import numpy as np
from atom3d.datasets import load_dataset
from collapse.utils import pdb_from_fname
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Script for searching database given query residue')
parser.add_argument('db', type=str, help='pre-computed embedding database in LMDB format')
parser.add_argument('outfile', type=str, help='Output pkl file')
args = parser.parse_args()

database = load_dataset(args.db, 'lmdb')
    
e_db = []
pdb_ids = []
chains = []
resids = []
for elem in tqdm(database):
    if elem is None:
        continue
    name, _ = pdb_from_fname(elem['id'])
    chains.extend(elem['chains'])
    resids.extend(elem['resids'])
    pdb_ids.extend([name] * len(elem['resids']))
    e_db.append(elem['embeddings'])

e_db = np.concatenate(e_db, 0)

print(e_db.shape)

outdata = {'embeddings': e_db, 'pdbs': pdb_ids, 'chains': chains, 'resids': resids}

with open(args.outfile, 'wb') as f:
    pickle.dump(outdata, f)