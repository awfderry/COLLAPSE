from atom3d.datasets import load_dataset
from atom3d.filters.filters import first_model_filter
from collapse.data import MSA
from Bio import AlignIO
from collapse.atom_info import aa_to_letter
import os
import parallel as par
import subprocess
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='../data/datasets/cdd_af2_dataset')
args = parser.parse_args()

def get_chain_sequences(df):
    """Return list of tuples of (id, sequence) for different chains of monomers in a given dataframe."""
    # Keep only CA of standard residues
    df = df[df['name'] == 'CA'].drop_duplicates()
    # df = df[df['resname'].apply(lambda x: Poly.is_aa(x, standard=True))]
    df['resname'] = df['resname'].apply(aa_to_letter)
    chain_sequences = []
    chain_resids = []
    for c, chain in df.groupby(['ensemble', 'subunit', 'structure', 'model', 'chain']):
        cid = c[0].split('.')[0] + '_' + c[4]
        seq = ''.join(chain['resname'])
        resid = chain['resname']+chain['residue'].astype(str)
        chain_sequences.append((cid, seq))
        chain_resids.append((cid, resid.tolist()))
        assert len(seq)==len(resid)
    return chain_sequences, chain_resids

def write_fasta(chain_sequences, outfile):
    """
    Write chain_sequences to fasta file. 
    """
    with open(outfile, 'w') as f:
        for chain, seq in chain_sequences:
            f.write('>' + chain + '\n')
            f.write(seq + '\n')

def cdd_transform(item):
    atoms = item['atoms']
    if len(atoms) == 0:
        return None
    atoms = first_model_filter(atoms)
    atoms = atoms[~atoms.hetero.str.contains('W')]
    item['atoms'] = atoms[atoms.element != 'H'].reset_index(drop=True)
    # item['atoms'] = atoms[atoms['hetero'].str.strip()=='']
    return item

def align_pdb_to_msa(cdd_id):
    if os.path.exists(f'../data/msa_pdb_aligned/{cdd_id}.afa'):
        return
    
    path = os.path.join('../data/msa', cdd_id + '.FASTA')
    with open(path) as f:
        alignment = AlignIO.read(f, "fasta")
    msa = MSA(alignment)
    if len(msa) <= 1:
        return

    with open(f'../data/msa_pdb_aligned/{cdd_id}.afa', 'w') as f:
        f.write(msa.__format__('fasta'))

    for record in msa:
        split = record.id.split('|')
        if '|pdb|' in record.id:
            if split[3].strip() == '':
                continue
            elif (len(split) < 5) or (split[4].strip() == ''):
                pdb_chain = split[3].lower() + '_A'
            else:
                pdb_chain = split[3].lower() + '_' + split[4]
        elif '|sp|' in record.id:
            if split[3].strip() == '':
                continue
            pdb_chain = record.id.split('|')[3] + '_A'
        ident = pdb_chain.split('_')[0]
        if ident not in pdb_to_idx:
            continue
        dataset_idx = pdb_to_idx[ident]
        atoms = pdb_dataset[dataset_idx]['atoms']
        chain_sequences, resids = get_chain_sequences(atoms)
        chain_seq_filt = [(c, s) for c, s in chain_sequences if c == pdb_chain]
        write_fasta(chain_seq_filt, f'../data/fasta/{pdb_chain}.fasta')
        subprocess.call(f'./muscle -profile -in1 ../data/msa_pdb_aligned/{cdd_id}.afa -in2 ../data/fasta/{pdb_chain}.fasta -out ../data/msa_pdb_aligned/{cdd_id}.afa'.split())

with open('pdb_resid.json', 'r') as fout:
    pdb_to_resid = json.load(fout)
with open('pdb_idx.json', 'r') as fout:
    pdb_to_idx = json.load(fout)

cdd_dataset_ = load_dataset(args.dataset, 'lmdb')
cdd_set = set(cdd_dataset_.ids())

pdb_dataset = load_dataset('../data/cdd_pdb_sp', 'pdb.gz', transform=cdd_transform)

inputs = [tuple([c]) for c in cdd_set]
par.submit_jobs(align_pdb_to_msa, inputs, num_threads=16)