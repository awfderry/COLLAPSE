import numpy as np
import os
import argparse
import random
import pickle
from tqdm import tqdm
import torch
import collections as col
from collapse.data import PDBSampleDataset
import atom3d.util.file as fi
from torch_geometric.loader import DataLoader
from collapse import initialize_model
from scipy.spatial.distance import cosine
from collapse import atom_info

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str)
parser.add_argument('out_dir', type=str)
parser.add_argument('--checkpoint', type=str, default='data/checkpoints/collapse_base.pt')
parser.add_argument('--num_samples', type=int, default=1000000)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

all_files = fi.find_files(args.data_dir, 'pdb')

dataset = PDBSampleDataset(args.data_dir, train_mode=False)
print(f'Processing dataset with {len(all_files)} examples...')

loader = DataLoader(dataset, batch_size=4, shuffle=False)

model = initialize_model(args.checkpoint, device=device)

loader = DataLoader(dataset, batch_size=1, shuffle=False)

print('Computing embeddings...')
# Save embeddings to files by PDB
restype_to_emb = col.defaultdict(list)
with torch.no_grad():
    for g, resname in tqdm(loader):
        g = g.to(device)
        embeddings, _ = model.online_encoder(g, return_projection=False)
        restype_to_emb[resname[0]].append(embeddings.squeeze().cpu().numpy())


print('Saving to files...')
for resname, emb in restype_to_emb.items():
    print(resname, len(emb))
    emb = np.stack(emb, 0)
    outfile = os.path.join(args.out_dir, f'{resname}')
    np.save(outfile, emb)

restype_to_emb = {}
for resname in atom_info.aa[:20]:
    restype_to_emb[resname] = np.load(os.path.join(args.out_dir, f'{resname}.npy'))

def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)

print('Computing background...')
cosines = col.defaultdict(list)
for resname, emb in restype_to_emb.items():
    print(resname)
    rows = range(len(emb))
    for _ in range(args.num_samples):
        r1, r2 = random_combination(rows, 2)
        cos = 1 - cosine(emb[r1], emb[r2])
        cosines[resname].append(cos)
        cosines['combined'].append(cos)

pvals = [1e-2, 9e-3, 8e-3, 7e-3, 6e-3, 5e-3, 4e-3, 3e-3, 2e-3, 1e-3, 9e-4, 8e-4, 7e-4, 6e-4, 5e-4, 4e-4, 3e-4, 2e-4, 1e-4, 9e-5, 8e-5, 7e-5, 6e-5, 5e-5, 4e-5, 3e-5, 2e-5, 1e-5]
quants = [1 - p for p in pvals]

for resname, cos_list in cosines.items():
    quantiles = np.quantile(cos_list, quants)
    mean_cos = np.mean(cos_list)
    std_cos = np.std(cos_list)
    data = {pvals[i]: quantiles[i] for i in range(len(quantiles))}
    data['mean'] = mean_cos
    data['std'] = std_cos
    with open(f'../data/background_stats/{resname}_background.pkl', 'wb') as f:
        pickle.dump(data, f)
    with open(f'../data/background_stats/{resname}_background_dist.pkl', 'wb') as f:
        pickle.dump(cos_list, f)
