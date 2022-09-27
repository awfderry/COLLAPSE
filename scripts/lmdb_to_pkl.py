import argparse
import numpy as np
from atom3d.datasets import load_dataset
from collapse.utils import pdb_from_fname, contiguous_high_confidence_regions
import pickle
from tqdm import tqdm
import torch

parser = argparse.ArgumentParser(description='Script for converting LMDB embedding dataset to pickle')
parser.add_argument('db', type=str, help='pre-computed embedding database in LMDB format')
parser.add_argument('out_path', type=str, help='Output pkl file')
parser.add_argument('--split_id', type=int, default=0)
parser.add_argument('--num_splits', type=int, default=1)
parser.add_argument('--confidence', action='store_true')
args = parser.parse_args()

dataset = load_dataset(args.db, 'lmdb')

if args.num_splits > 1:
    out_path = args.out_path.replace('.pkl', f'_{args.split_id}.pkl')

    split_idx = np.array_split(np.arange(len(dataset)), args.num_splits)[args.split_id - 1]
    print(f'Processing split {args.split_id} with {len(split_idx)} examples...')

    dataset = torch.utils.data.Subset(dataset, split_idx)
else:
    out_path = args.out_path
    print(f'Processing full dataset with {len(dataset)} examples...')
    
e_db = []
pdb_ids = []
chains = []
resids = []
for elem in tqdm(dataset):
    if elem is None:
        continue
    name, _ = pdb_from_fname(elem['id'])
    contig = contiguous_high_confidence_regions(elem['confidence'], 20)
    if len(contig) == 0:
        continue
    chains.extend(list(np.array(elem['chains'])[contig]))
    resids.extend(list(np.array(elem['resids'])[contig]))
    pdb_ids.extend([name] * len(contig))
    e_db.append(elem['embeddings'][contig, :])

e_db = np.concatenate(e_db, 0)

print(e_db.shape)

outdata = {'embeddings': e_db, 'pdbs': pdb_ids, 'chains': chains, 'resids': resids}

with open(out_path, 'wb') as f:
    pickle.dump(outdata, f, protocol=4)