import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
import torch
import pickle
from collapse.data import process_pdb, initialize_model, embed_protein

def download_pdb(pdb, data_dir, assembly=1):
    if assembly:
        f = data_dir + "/" + pdb + ".pdb1"
        if not os.path.isfile(f):
            try:
                os.system("wget -q -O {}.gz https://files.rcsb.org/download/{}.pdb1.gz".format(f, pdb.upper()))
                os.system("gunzip {}.gz".format(f))

            except:
                f = data_dir + "/" + pdb + ".pdb"
                if not os.path.isfile(f):
                    os.system("wget -O {} https://files.rcsb.org/download/{}.pdb".format(f, pdb.upper()))
    else:
        f = data_dir + "/" + pdb + ".pdb"

    if not os.path.isfile(f):
        os.system("wget -q -O {} https://files.rcsb.org/download/{}.pdb".format(f, pdb.upper()))

    return f

task = sys.argv[1]

data_dir = '../data/xin_functional_benchmarks/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

positives = pd.read_csv(os.path.join(data_dir, f'{task}.pos'), sep='\t', names=['pdb', 'chain', 'residue'])
negatives = pd.read_csv(os.path.join(data_dir, f'{task}.neg'), sep='\t', names=['pdb', 'chain', 'residue'])
print(f'{len(positives)} positive examples, {len(negatives)} negative examples')
dataset = pd.concat([positives, negatives])
dataset['pdb_chain'] = dataset['pdb'].str.lower() + '_' + dataset['chain']
dataset['label'] = [1]*len(positives) + [0]*len(negatives)

chain_to_fold = {}
for fold, fold_chains in enumerate(np.array_split(dataset.pdb_chain.unique(), 10)):
    for c in fold_chains:
        chain_to_fold[c] = fold

model = initialize_model(checkpoint='../data/checkpoints/byol-radius-10.0-cutoff-4.5.pt', device=device)

for p in dataset.pdb_chain.unique():
    pdb, chain = p.split('_')
    if not os.path.exists(os.path.join(data_dir, 'pdb', pdb+'.pdb')):   
        if os.path.exists(os.path.join('/oak/stanford/groups/rbaltman/aderry/pdb/localpdb/mirror/pdb', p[1:3], 'pdb'+pdb+'.ent.gz')):
            in_dir = os.path.join('/oak/stanford/groups/rbaltman/aderry/pdb/localpdb/mirror/pdb', p[1:3], 'pdb'+pdb+'.ent.gz')
            out_dir = os.path.join(data_dir, 'pdb', pdb + '.pdb.gz')
            shutil.copy(in_dir, out_dir)
            os.system(f"gunzip {out_dir}")
        else:
            download_pdb(pdb, os.path.join(data_dir, 'pdb'), 0)
    
pdb_embeddings = {}
for p in tqdm(dataset.pdb_chain.unique(), desc='Embedding proteins'):
    pdb, chain = p.split('_')
    atom_df = process_pdb(os.path.join(data_dir, 'pdb', pdb+'.pdb'), chain)
    embedding_data = embed_protein(atom_df, model, device, include_hets=False)
    if embedding_data is None:
        print(f'skipping PDB {p}: too large for memory')
        continue
    pdb_embeddings[p] = dict(zip([r[1:] for r in embedding_data['resids']], embedding_data['embeddings']))

fold_data = {i:{'X':[], 'y':[]} for i in range(10)}
for r, row in tqdm(dataset.iterrows(), total=len(dataset), desc='Extracting residues'):
    if row['pdb_chain'] not in pdb_embeddings:
        continue
    emb_dict = pdb_embeddings[row['pdb_chain']]
    emb = emb_dict.get(str(row['residue']))
    if emb is None:
        continue
    fold = chain_to_fold[row['pdb_chain']]
    fold_data[fold]['X'].append(emb)
    fold_data[fold]['y'].append(row['label'])

for f, d in fold_data.items():
    fold_data[f]['X'] = np.stack(d['X'])

print('Saving dataset to disk...')
with open(os.path.join(data_dir, f'{task}_embeddings.pkl'), 'wb') as f:
    pickle.dump(fold_data, f)
    
