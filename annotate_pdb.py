import argparse
import numpy as np
import pickle
import torch
from fastdist import fastdist
from atom3d.datasets import load_dataset
from collapse.data import EmbedTransform
from collapse import initialize_model
from collapse.utils import pdb_from_fname
import collections as col


parser = argparse.ArgumentParser()
parser.add_argument('pdb', type=str, nargs='+')
parser.add_argument('--chains', type=str, default=None)
parser.add_argument('--db', type=str, default='data/datasets/full_site_db_stats.pkl')
parser.add_argument('--cutoff', type=float, default=1e-4)
parser.add_argument('--site_cutoff', type=float, default=1e-4)
parser.add_argument('--checkpoint', type=str, default='data/checkpoints/collapse_base.pt')
parser.add_argument('--filetype', type=str, default='pdb')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--include_hets', action='store_true')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(args.db, 'rb') as f:
    db = pickle.load(f)
db_embeddings = db['embeddings']
db_labels = np.array(db['sites'])
db_sources = np.array(db['sources'])
db_pdbs = np.array(db['pdbs'])
db_resids = np.array(db['resids'])
db_means = np.array(db['mean_cos'])
db_stds = np.array(db['std_cos'])
db_cutoffs = np.array([q[args.site_cutoff] for q in db['quantiles']])


print(f'Searching {len(args.pdb)} PDBs against database of size {len(db_pdbs)}, representing {len(set(db_labels))} functional sites')
    
with open('data/background_stats/combined_background.pkl', 'rb') as f:
    quants = pickle.load(f)
    cutoff = quants[args.cutoff]

model = initialize_model(args.checkpoint, device=device)

transform = EmbedTransform(model, include_hets=args.include_hets, device=device)
dataset = load_dataset(args.pdb, args.filetype, transform=transform)

db_pdbcodes = np.array([p[:4] for p in db_pdbs])

for pdb_data in dataset:
    pdb_id, af_flag = pdb_from_fname(pdb_data["id"])
    print(f'Input PDB: {pdb_id}')
    
    if pdb_id[:4] in db_pdbcodes:
        idx_to_remove = np.where(db_pdbcodes == pdb_id[:4])[0]
        db_pdbs = np.delete(db_pdbs, idx_to_remove)
        db_sources = np.delete(db_sources, idx_to_remove)
        db_labels = np.delete(db_labels, idx_to_remove)
        db_resids = np.delete(db_resids, idx_to_remove)
        db_embeddings = np.delete(db_embeddings, idx_to_remove, 0)
        db_means = np.delete(db_means, idx_to_remove, 0)
        db_stds = np.delete(db_stds, idx_to_remove, 0) 
        db_cutoffs = np.delete(db_cutoffs, idx_to_remove, 0) 
    
    resids = np.array(pdb_data['resids'])
    chains = np.array(pdb_data['chains'])
    embeddings = np.array(pdb_data['embeddings'])
    confidences = np.array(pdb_data['confidence'])
    
    if af_flag:
        print('Removing low confidence residues')
        high_conf_idx = confidences >= 70
        resids = resids[high_conf_idx]
        chains = chains[high_conf_idx]
        embeddings = embeddings[high_conf_idx]
    
    if args.chains:
        print(f'Annotating chains: {args.chains}')
        chain_idx = np.in1d(chains, np.array(list(args.chains)))
        resids = resids[chain_idx]
        chains = chains[chain_idx]
        embeddings = embeddings[chain_idx]
        
    
    cosines = fastdist.cosine_matrix_to_matrix(embeddings, db_embeddings)  # (n_res, n_db)
    
    query_mask = cosines > cutoff
    site_mask = cosines > db_cutoffs[np.newaxis, :]
    
    quantile_mask = query_mask & site_mask

    results = col.defaultdict(dict)
    hit_idx_by_row = [np.nonzero(row)[0] for row in quantile_mask]
    
    for i, hit_idx in enumerate(hit_idx_by_row):
        if len(hit_idx) == 0:
            continue
        hits = np.unique(hit_idx)
        chain_res = chains[i] + '_' + resids[i]
        for h in hits:
            key = (db_labels[h], db_sources[h])
            if chain_res in results[key]:
                results[key][chain_res].add(f'{db_pdbs[h]}: {db_resids[h]}')
            else:
                results[key][chain_res] = set([f'{db_pdbs[h]}: {db_resids[h]}'])
    
    print('Results at p = ', args.cutoff)
    for (name, source), sites in results.items():
        print(f' {name} ({source})')
        for loc, pdbs in sites.items():
            if args.verbose:
                print(f"    - {loc}: {pdbs}")
            else:
                print(f"    - {loc}: {len(pdbs)} PDBs")
