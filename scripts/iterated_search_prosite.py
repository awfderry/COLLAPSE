import os
import numpy as np
import pandas as pd
import argparse
import pickle
import collections as col
from collapse import atom_info
# from sklearn.metrics.pairwise import cosine_similarity
import time
from sklearn.preprocessing import MinMaxScaler
import faiss

label_defs = {'TP': 1, 'TN': 0, 'FN': 1, 'FP': 0}
np.random.seed(2)


def main(args):
    
    start = time.time()
    
    site_name = os.path.splitext(os.path.basename(args.site_db))[0]
    functional_residue = atom_info.prosite_residues[site_name]
    with open(f'background_stats/{functional_residue}_background.pkl', 'rb') as f:
        aa_stats = pickle.load(f)
    
    if args.cutoff is not None:
        cutoff_list = [args.cutoff]
    else:
        cutoff_list = [0.995, 0.999, 0.9995, 0.9999, 0.99999]
        percentiles = ['5e-3', '1e-3', '5e-4', '1e-4', '1e-5']

    with open(args.site_db, 'rb') as datafile:
        db_data = pickle.load(datafile)
    e_p = db_data['embeddings'] # (M, 512)
    faiss.normalize_L2(e_p)
    
    d = e_p.shape[1]
    index = faiss.IndexFlatIP(d)  # the other index
    index.train(e_p)
    index.add(e_p)
    
    if args.normalize:
        scaler = MinMaxScaler()
        e_p = scaler.fit_transform(e_p)
    
    pdb_ids = np.array(db_data['pdbs'])
    labels = np.array(db_data['labels'])
    bin_labels = np.array([label_defs[lab] for lab in labels])
    
    tp_idx = np.where(labels == 'TP')[0]
    fp_pdbs = set(pdb_ids[np.where(labels == 'FP')[0]])
    fn_pdbs = set(pdb_ids[np.where(labels == 'FN')[0]])
    
    print('Site:', site_name)
    print(f'Total hits in database = {np.sum(bin_labels)}')
    
    results_df = col.defaultdict(list)
    
    for samp in range(args.num_sample):
        rand_idx = np.random.choice(tp_idx, size=args.num_query, replace=False)
        for i, cutoff in enumerate(cutoff_list):
            # print('Testing cutoff', cutoff)
            eff_cutoff = aa_stats[cutoff]
            e_q = e_p[rand_idx] # (N, 512)
            print('Query PDB:', pdb_ids[rand_idx], 'Cutoff:', cutoff)
            result_idx = set()
            for it in range(args.num_iter):
                # print(eff_cutoff)
                n_query = e_q.shape[0]
                start = time.time()
                _, _, idx = index.range_search(e_q, eff_cutoff)
                elapsed = time.time() - start
                idx = np.unique(idx)
                idx = idx[(idx < index.ntotal) & (idx > 0)]
                if len(idx) == 0:
                    continue
                e_q = e_p[idx]
                # print(f'Iteration {it+1}: PBDs {list(zip(pdb_ids[sorted_idx], sorted_scores, labels[sorted_idx], bin_labels[sorted_idx]))}')
                result_idx = set(idx).union(result_idx)
                hits = bin_labels[np.array(list(result_idx), dtype=int)]
                recall = float(np.sum(hits)) / np.sum(bin_labels)
                precision = float(np.sum(hits)) / len(result_idx)
                tp_hits = result_idx.intersection(set(tp_idx))
                tp_recall = float(len(tp_hits)) / len(tp_idx)
                hit_pdbs = set(pdb_ids[np.array(list(result_idx), dtype=int)])
                fp_pdb_hits = hit_pdbs.intersection(fp_pdbs)
                fn_pdb_hits = hit_pdbs.intersection(fn_pdbs)
                
                if args.verbose:
                    print(f'Sample {samp+1}, Iteration: {it+1}')
                    print('\tNumber of results:', len(result_idx))
                    print('\tNumber of hits:', np.sum(hits))
                    print(f'\tRecall TP: {len(tp_hits)} / {len(tp_idx)}')
                    print(f'\tRecall TP + FN: {np.sum(hits)} / {np.sum(bin_labels)}')
                    print(f'\tFP PDB hits: {len(fp_pdb_hits)} / {len(fp_pdbs)}')
                    print(f'\tFN PDB hits: {len(fn_pdb_hits)} / {len(fn_pdbs)}')
                    print('\tPrecision:', precision)
                results_df['iteration'].append(it + 1)
                results_df['recall'].append(recall)
                results_df['TP correct'].append(tp_recall)
                results_df['FN correct'].append(float(len(fn_pdb_hits)) / len(fn_pdbs))
                results_df['FP correct'].append(float(len(fp_pdb_hits)) / len(fp_pdbs))
                results_df['precision'].append(precision)
                results_df['cutoff'].append(percentiles[i])
                results_df['iter_time'].append(elapsed)
                results_df['n_query'].append(n_query)
    
    results_df = pd.DataFrame(results_df)
    results_df.to_csv(args.outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for searching database given query residues')
    parser.add_argument('site_db', type=str, help='directory containing pre-computed embeddings for protein database')
    parser.add_argument('outfile', type=str, help='file to save results')
    parser.add_argument('--checkpoint', type=str, default='data/checkpoints/collapse_base.pt')
    parser.add_argument('--num_query', type=int, default=1, help='number of queries to sample')
    parser.add_argument('--num_iter', type=int, default=5, help='number of search iterations')
    parser.add_argument('--num_sample', type=int, default=10, help='number of samples')
    parser.add_argument('--cutoff', type=float, default=None, help='similarity cutoff for inclusion at each iteration')
    parser.add_argument('--verbose', action='store_true', help='whether to print output')
    parser.add_argument('--normalize', action='store_true', help='min-max normalize inputs (for use with FEATURE)')

    args = parser.parse_args()
    
    main(args)
