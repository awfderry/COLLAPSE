import os
import torch
import numpy as np
import pandas as pd
import argparse
import pickle
import collections as col
from collapse import atom_info, initialize_model, process_pdb, embed_residue
from collapse.utils import pdb_from_fname, quantile_from_score
from atom3d.datasets import load_dataset
from tqdm import tqdm
import faiss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    query_pdb, _ = pdb_from_fname(args.pdb)
    chain_resid = (args.chain, args.resid)
    atom_df = process_pdb(args.pdb, chain=args.chain)
    model = initialize_model(args.checkpoint, device=device)
    e_q = embed_residue(atom_df, chain_resid, model, device=device, include_hets=args.include_hets)
    if len(e_q.shape) == 1:
        e_q = e_q[np.newaxis, :]
    
    with open(f'data/background_stats/{atom_info.letter_to_aa(args.resid[0])}_background.pkl', 'rb') as f:
        aa_stats = pickle.load(f)
    eff_cutoff = aa_stats[args.cutoff]
    
    with open(args.db, 'rb') as datafile:
        db_data = pickle.load(datafile)
    e_db = db_data['embeddings'] # (M, 512)
    pdb_ids = np.array(db_data['pdbs'])
    resids = np.array(db_data['resids'])
    if len(pdb_ids[0].split('_')[0]) > 4:
        af_flag = True
    
    pdb_meta = pd.read_csv(args.metadata, index_col=0, sep=None)
    
    if query_pdb not in pdb_meta.index:
        pdb_meta = pdb_meta.append(pd.Series(data=['N/A'] * pdb_meta.shape[1], index=pdb_meta.columns, name=query_pdb))
    
    # filter DB to same residue, for efficiency
    residues = np.array([r[0] for r in db_data['resids']])
    res_mask = residues == args.resid[0]
    e_db = e_db[res_mask]
    pdb_ids = pdb_ids[res_mask]
    resids = resids[res_mask]
    
    print('Database size:', len(e_db))
    
    faiss.normalize_L2(e_db)
    d = e_db.shape[1]
    index = faiss.IndexFlatIP(d) 
    index.train(e_db)
    index.add(e_db)
    
    query_set = e_q.copy()
    query_pdb = [query_pdb + '_' + args.chain]
    faiss.normalize_L2(query_set)
    
    results = {'PDB': query_pdb, 'RESID': [args.resid], 'COSINE': [1.0], 'ITER': [0], 'QUERY': ['N/A']}
    
    used = set()
    for it in range(args.num_iter):
        lims, dists, idx = index.range_search(query_set, eff_cutoff)
        all_idx = np.unique(idx)
        new = np.array([x for x in all_idx if x not in used])
        used.update(set(all_idx))
        if len(idx) == 0:
            print('No results found!')
            continue
        query_set = e_db[new]
        
        if args.verbose:
            print(f'Iteration {it + 1}: {len(new)} new results')
        
        for q_i in range(len(lims) - 1):
            dists_i = dists[lims[q_i]:lims[q_i + 1]]
            idx_i = idx[lims[q_i]:lims[q_i + 1]]
            pdbs_i = pdb_ids[idx_i]
            resids_i = resids[idx_i]
            # print(f'query {q_i}: {len(dists_i)} neighbors -- {pdbs_i}')
            results['PDB'].extend(pdbs_i.tolist())
            results['RESID'].extend(resids_i.tolist())
            results['COSINE'].extend(quantile_from_score(dists_i))
            results['ITER'].extend([it + 1] * len(idx_i))
            results['QUERY'].extend([query_pdb[q_i]] * len(idx_i))
        query_pdb = pdb_ids[new].tolist()
    

    results = pd.DataFrame(results)
    results = results.drop_duplicates(subset=['PDB'])
    
    cols = ['Description', 'Classification', 'Keywords', 'Method', 'Uniprot', 'Citation']
    
    # results[cols] = results['PDB'].apply(lambda x: pdb_meta.loc[x[:4], cols])
    results['pdb4'] = results['PDB'].str[:4]
    results = pd.merge(results, pdb_meta.loc[:, cols], left_on='pdb4', right_index=True, how='left')
    
    if args.verbose:
        print(results)
    
    results.to_csv(args.outfile)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for searching database given query residue')
    parser.add_argument('pdb', type=str, help='Input PDB file')
    parser.add_argument('chain', type=str, help='Chain of query residue')
    parser.add_argument('resid', type=str, help='Query residue ID in letter+resnum format (e.g. A42)')
    parser.add_argument('db', type=str, help='pre-computed embedding database in pickle format')
    parser.add_argument('--outfile', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default='data/checkpoints/collapse_base.pt')
    parser.add_argument('--metadata', type=str, default='data/mappings/pdb_metadata.csv')
    parser.add_argument('--num_iter', type=int, default=3, help='number of search iterations')
    parser.add_argument('--cutoff', type=float, default=1e-4, help='similarity cutoff for inclusion at each iteration')
    parser.add_argument('--verbose', action='store_true', help='whether to print output')
    parser.add_argument('--include_hets', action='store_true')
    
    args = parser.parse_args()
    
    if not args.outfile:
        args.outfile = f'{pdb_from_fname(args.pdb)}_{args.chain}_{args.resid}_search.csv'
    
    main(args)
