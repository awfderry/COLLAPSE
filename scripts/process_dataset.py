import os
from Bio import AlignIO
from data import MSA
from tqdm import tqdm
import pandas as pd
import json
from atoms import aa_to_letter
import parallel as par

from atom3d.datasets import load_dataset, make_lmdb_dataset
from atom3d.filters.filters import first_model_filter
import subprocess

oak_dir = '/oak/stanford/groups/rbaltman/aderry'

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
        resid = chain['resname'] + chain['residue'].astype(str)
        chain_sequences.append((cid, seq))
        chain_resids.append((cid, resid.tolist()))
        assert len(seq) == len(resid)
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

pdb_dataset = load_dataset(os.path.join(oak_dir, 'COLLAPSE', 'cdd_assembly_new'), 'pdb', transform=cdd_transform)

pdb_to_idx = {}
pdb_to_resid = {}
lengths = []
for i, item in enumerate(tqdm(pdb_dataset)):
    ident = item['id'].split('.')[0]
    pdb_to_idx[ident] = i
    try:
        _, resid = get_chain_sequences(item['atoms'])
    except:
        print(ident)
        continue
    pdb_to_resid[ident] = dict(resid)
    lengths.append(len(item['atoms']))
    
with open('pdb_resid.json', 'w') as fout:
    json.dump(pdb_to_resid, fout)
with open('pdb_idx.json', 'w') as fout:
    json.dump(pdb_to_idx, fout)

# with open('pdb_resid.json', 'r') as fin:
#     pdb_to_resid = json.load(fin)
# with open('pdb_idx.json', 'r') as fin:
#     pdb_to_idx = json.load(fin)


files = [x for x in os.listdir('msa') if x[:2] == 'cd']
cdd_dataset = []
for i, fa in enumerate(tqdm(files)):
    path = os.path.join('msa', fa)
    with open(path) as f:
        alignment = AlignIO.read(f, "fasta")
    msa = MSA(alignment)

    if len(msa) <= 1:
        continue
    atoms_list = []
    pdb_ids = []
    resids = []
    for i, record in enumerate(msa):
        if 'pdb' in record.id:
            ident = record.id.split('|')[-2].lower()
            pdb_chain = ident + '_' + record.id.split('|')[-1]
            if len(pdb_chain) == 5:
                pdb_chain = pdb_chain + 'A'
            elif len(pdb_chain) > 6:
                continue
        else:
            ident = record.id.split('|')[-2]
            pdb_chain = ident + '_A'
        
        if ident in pdb_to_resid:
            ri = pdb_to_resid[ident].get(pdb_chain)
        else:
            continue
        if ri is not None:
            resids.append(ri)
        else:
            continue
        pdb_ids.append(pdb_chain)
        dataset_idx = pdb_to_idx[ident]
        atoms = pdb_dataset[dataset_idx]['atoms']
        if (atoms is None):
            continue

        atoms_list.append(atoms)
    if len(atoms_list) > 1:
        atoms_combined = pd.concat(atoms_list)
    else:
        continue
    item = {'id': fa[:-6],
            'alignment': alignment,
            'atoms': atoms_combined,
            'pdb_ids': pdb_ids,
            'residue_ids': resids}
    cdd_dataset.append(item)
print(f'Number of families in processed dataset: {len(cdd_dataset)}')

make_lmdb_dataset(cdd_dataset, 'lmdb/cdd_af2_dataset', serialization_format='pkl', filter_fn=lambda x: x is None)


cdd_dataset = load_dataset('lmdb/cdd_af2_dataset', 'lmdb')
cdd_set = [tuple([c]) for c in cdd_dataset.ids()]

def process_msa(cdd_id):
    path = os.path.join('msa', cdd_id + '.FASTA')
    with open(path) as f:
        alignment = AlignIO.read(f, "fasta")
    msa = MSA(alignment)
    if len(msa) <= 1:
        return

    subprocess.call(f'cp {path} msa_pdb_aligned/{cdd_id}.afa'.split())
    for record in msa:
        if 'pdb' in record.id:
            pdb = record.id.split('|')[-2].lower() + '_' + record.id.split('|')[-1]
            if len(pdb) == 4:
                pdb = pdb + '_A'
        elif 'sp' in record.id:
            pdb = record.id.split('|')[-2] + '_A'
        ident = pdb.split('_')[0]
        if ident not in pdb_to_idx:
            return
        dataset_idx = pdb_to_idx[ident]
        atoms = pdb_dataset[dataset_idx]['atoms']
        chain_sequences, resids = get_chain_sequences(atoms)
        chain_seq_filt = [(c, s) for c, s in chain_sequences if c == pdb]
        write_fasta(chain_seq_filt, f'fasta/{pdb}_pdb.fasta')
        subprocess.call(f'./muscle -profile -in1 msa_pdb_aligned/{cdd_id}.afa -in2 fasta/{pdb}_pdb.fasta -out msa_pdb_aligned/{cdd_id}.afa'.split())

for c in tqdm(cdd_dataset.ids()):
    process_msa(c)
    
# par.submit_jobs(process_msa, cdd_set, 8)
