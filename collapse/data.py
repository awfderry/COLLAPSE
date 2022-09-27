import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import os
import torch
import math
import random
import pickle
from torch.utils.data import IterableDataset
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from atom3d.datasets import load_dataset
import atom3d.util.formats as fo
from atom3d.filters.filters import first_model_filter
import collapse.utils as nb
import scipy.spatial
import collections as col
from scipy.stats import entropy
from collapse import atom_info
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData, Data
import torch_cluster
from collections.abc import Mapping, Sequence
from collapse.byol_pytorch import BYOL
from collapse.models import CDDModel

import pathlib
DATA_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), '../data')
np.random.seed(2)

"""
Define parameters and standard transforms for converting atom environments to graphs (from GVP). 
"""
# =========================

_NUM_ATOM_TYPES = 13
_element_mapping = lambda x: {
    'C': 0,
    'N': 1,
    'O': 2,
    'F': 3,
    'S': 4,
    'Cl': 5, 'CL': 5,
    'P': 6,
    'Se': 7, 'SE': 7,
    'Fe': 8, 'FE': 8,
    'Zn': 9, 'ZN': 9,
    'Ca': 10, 'CA': 10,
    'Mg': 11, 'MG': 11,
}.get(x, 12)

def _normalize(tensor, dim=-1):
    '''
    Adapted from https://github.com/drorlab/gvp-pytorch
     
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    and 
    https://github.com/drorlab/gvp-pytorch
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF
    
def _edge_features(coords, edge_index, D_max=4.5, num_rbf=16, device='cpu'):
    """Adapted from https://github.com/drorlab/gvp-pytorch"""
    
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), 
               D_max=D_max, D_count=num_rbf, device=device)

    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))

    return edge_s, edge_v

class BaseTransform:
    '''
    Adapted from https://github.com/drorlab/gvp-pytorch
    
    Implementation of an ATOM3D Transform which featurizes the atomic
    coordinates in an ATOM3D dataframes into `torch_geometric.data.Data`
    graphs. This class should not be used directly; instead, use the
    task-specific transforms, which all extend BaseTransform. Node
    and edge features are as described in the EGNN manuscript.
    
    Returned graphs have the following attributes:
    -x          atomic coordinates, shape [n_nodes, 3]
    -atoms      numeric encoding of atomic identity, shape [n_nodes]
    -edge_index edge indices, shape [2, n_edges]
    -edge_s     edge scalar features, shape [n_edges, 16]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]
    
    Subclasses of BaseTransform will produce graphs with additional 
    attributes for the tasks-specific training labels, in addition 
    to the above.
    
    All subclasses of BaseTransform directly inherit the BaseTransform
    constructor.
    
    :param edge_cutoff: distance cutoff to use when drawing edges
    :param num_rbf: number of radial bases to encode the distance on each edge
    :device: if "cuda", will do preprocessing on the GPU
    '''
    def __init__(self, edge_cutoff=4.5, num_rbf=16, device='cpu'):
        self.edge_cutoff = edge_cutoff
        self.num_rbf = num_rbf
        self.device = device
            
    def __call__(self, df):
        '''
        :param df: `pandas.DataFrame` of atomic coordinates
                    in the ATOM3D format
        
        :return: `torch_geometric.data.Data` structure graph
        '''
        with torch.no_grad():
            coords = torch.as_tensor(df[['x', 'y', 'z']].to_numpy(),
                                     dtype=torch.float32, device=self.device)
            atoms = torch.as_tensor(list(map(_element_mapping, df.element)), dtype=torch.long, device=self.device)

            edge_index = torch_cluster.radius_graph(coords, r=self.edge_cutoff)

            edge_s, edge_v = _edge_features(coords, edge_index, D_max=self.edge_cutoff, num_rbf=self.num_rbf, device=self.device)
            
            data = Data(x=coords, atoms=atoms,
                        edge_index=edge_index, edge_s=edge_s, edge_v=edge_v)
            if 'same_chain' in df.columns:
                data.chain_ind = torch.as_tensor(df.same_chain.tolist(), dtype=torch.long, device=self.device)

            return data

"""
Instantiate general graph transform for all environment processing.
"""    
transform = BaseTransform()

# =========================

"""
Utility functions for processing environments
"""

def process_pdb(pdb_file, chain=None):
    atoms = fo.bp_to_df(fo.read_any(pdb_file))
    atoms = first_model_filter(atoms)
    if chain:
        atoms = atoms[atoms.chain == chain]
    atoms = atoms[~atoms.hetero.str.contains('W')]
    atoms = atoms[atoms.element != 'H'].reset_index(drop=True)
    return atoms

def initialize_model(checkpoint=os.path.join(DATA_DIR, 'checkpoints/collapse_base.pt'), train=False, device='cpu'):
    dummy_graph = torch.load(os.path.join(DATA_DIR, 'dummy_graph.pt'))
    model = BYOL(
        CDDModel(out_dim=512, scatter_mean=True, attn=False, chain_ind=False),
        projection_size=512,
        dummy_graph=dummy_graph,
        hidden_layer = -1,
        use_momentum = True,
        dense=False
    ).to(device)
    cpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(cpt['model_state_dict'])
    if train:
        model.train()
    else:
        model.eval()
    return model

def embed_protein(atom_df, model, device='cpu', include_hets=True, env_radius=10.0):
    emb_data = col.defaultdict(list)
    graphs = []
    if not include_hets:
        atom_df = atom_df[atom_df.resname.isin(atom_info.aa)].reset_index(drop=True)
    for (c, r, i), res_df in atom_df.groupby(['chain', 'resname', 'residue']):
        if r not in atom_info.aa[:20]:
            continue
        emb_data['chains'].append(c)
        resid = atom_info.aa_to_letter(r) + str(i)
        chain_atoms = atom_df[atom_df.chain == c]
        out = extract_env_from_resid(chain_atoms, (c, resid), env_radius, res_df.copy(), train_mode=False)
        if out is None:
            continue
        graphs.append(out)
        emb_data['resids'].append(resid)
        confidence = res_df['bfactor'].iloc[0]  # for AlphaFold pLDDT
        emb_data['confidence'].append(confidence)
    graphs = Batch.from_data_list(graphs).to(device)
    with torch.no_grad():
        try:
            embs, _ = model.online_encoder(graphs, return_projection=False)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): raise(e)
            torch.cuda.empty_cache()
            print('Out of Memory error!', flush=True)
            return None
    emb_data['embeddings'] = np.stack(embs.cpu().numpy(), 0)
    return emb_data

def embed_residue(atom_df, chain_resid, model, device='cpu', include_hets=True, env_radius=10.0):
    if not include_hets:
        atom_df = atom_df[atom_df.resname.isin(atom_info.aa)].reset_index(drop=True)
    chain, resid = chain_resid
    chain_atoms = atom_df[atom_df.chain == chain]
    out = extract_env_from_resid(chain_atoms, (chain, resid), env_radius, train_mode=False)
    graphs = Batch.from_data_list([out]).to(device)
    with torch.no_grad():
        emb, _ = model.online_encoder(graphs, return_projection=False)
    emb = emb.squeeze().cpu().numpy()
    return emb

def sample_functional_center(df, resid, train_mode=True):
    func_atoms = atom_info.abbr_key_atom_dict[resid[0]]
    if train_mode:
        func_atoms = func_atoms[np.random.choice(len(func_atoms), replace=False)]
    else:
        func_atoms = sum(func_atoms, [])
    
    coords = df[df['name'].isin(func_atoms)][['x', 'y', 'z']].astype(np.float32).to_numpy()
    if len(coords) == 0:
        coords = df[df['name'] == 'CA'][['x', 'y', 'z']].astype(np.float32).to_numpy()
    center = np.mean(coords, axis=0)
    return center

def extract_env_from_resid(df, ch_resid, env_radius, res_df=None, train_mode=False):
    chain, resid = ch_resid
    if resid[0] == 'X':
        print('Nonstandard residue')
        return None
    if res_df is None:
        df['resname'] = df['resname'].apply(atom_info.aa_to_letter)
        rows = (df['chain'] == chain) & (df['resname'] == resid[0]) & (df['residue'] == int(resid[1:]))
        res_df = df[rows]
    center = sample_functional_center(res_df, resid, train_mode)
        
    df = df.reset_index()
    kd_tree = scipy.spatial.cKDTree(df[['x', 'y', 'z']].to_numpy())

    pt_idx = kd_tree.query_ball_point(center, r=env_radius, p=2.0)
    df_env = df.iloc[pt_idx, :]
    
    if len(df_env) == 0:
        print('No environment found')
        return None
    
    graph = transform(df_env)
    
    return graph

def extract_env_from_coords(df, center, env_radius=10.0):
    df = df.reset_index()
    kd_tree = scipy.spatial.cKDTree(df[['x', 'y', 'z']].to_numpy())
    pt_idx = kd_tree.query_ball_point(center, r=env_radius, p=2.0)
    df_env = df.iloc[pt_idx, :]
    
    if len(df_env) == 0:
        return None
    
    graph = transform(df_env)
    return graph

# =========================

"""
Transforms and Pytorch Datasets for various situations
"""

class CDDTransform(object):
    '''
    Transforms LMDB dataset entries to featurized graphs. Returns a `torch_geometric.data.Data` graph
    '''
    
    def __init__(self, env_radius=10.0, single_chain=False, device='cpu'):
        self.env_radius = env_radius
        self.single_chain = single_chain
        self.device = device
    
    def __call__(self, elem, num_pairs_sampled=1):
        # pdbids = [p.replace('_', '') for p in elem['pdb_ids']]
        pdb_idx = dict(zip(elem['pdb_ids'], range(len(elem['pdb_ids']))))
        cdd_id = elem['id'] 
        
        with open(os.path.join(DATA_DIR, f'msa_pdb_aligned/{cdd_id}.afa')) as f:
            msa = MSA(AlignIO.read(f, 'fasta'))
        # msa = elem['msa']
        try:
            r1, r2, seq_r1, seq_r2 = msa.sample_record_pair()
        except:
            print('failed for MSA', cdd_id)
            return (None, None), None
        pair_ids = [r.id for r in (seq_r1, seq_r2)]

        pair_resids = [elem['residue_ids'][pdb_idx[p]] for p in pair_ids]
        
        pos1, pos2, cons = msa.sample_position_pairs(r1, r2, seq_r1, seq_r2, num_pairs=num_pairs_sampled)
        resid1, resid2 = pair_resids[0][pos1], pair_resids[1][pos2]
        chain1, chain2 = pair_ids[0][-1], pair_ids[1][-1]
        
        atoms = elem['atoms']
        df1, df2 = self._process_dataframes(atoms, pair_ids)
        graph1 = extract_env_from_resid(df1, (chain1, resid1), self.env_radius, train_mode=True)
        graph2 = extract_env_from_resid(df2, (chain2, resid2), self.env_radius, train_mode=True)
        
        metadata = {
            'res_labels': (atom_info.aa_to_label(resid1[0]), atom_info.aa_to_label(resid2[0])),
            'res_ids': (resid1, resid2),
            'pdb_ids': pair_ids,
            'cdd_id': cdd_id,
            'conservation': cons
        }
        
        return (graph1, graph2), metadata
    
    def _process_dataframes(self, atoms, pair):
        id1, id2 = pair
        id1_id, id1_chain = id1.split('_')
        id2_id, id2_chain = id2.split('_')

        if self.single_chain:
            df1 = atoms[(atoms['ensemble'].str.split('.').str[0] == id1_id) & (atoms['chain'] == id1_chain)]
            df2 = atoms[(atoms['ensemble'].str.split('.').str[0] == id2_id) & (atoms['chain'] == id2_chain)]
        else:
            df1 = atoms[(atoms['ensemble'].str.split('.').str[0] == id1_id)]
            df1['same_chain'] = (df1['chain'] == id1_chain).astype(int)
            df2 = atoms[(atoms['ensemble'].str.split('.').str[0] == id2_id)]
            df2['same_chain'] = (df2['chain'] == id2_chain).astype(int)

        return df1, df2


class EmbedTransform(object):
    '''
    Transforms LMDB PDBDataset entries
    to featurized graphs. Returns a `torch_geometric.data.Data`
    graph
    '''
    
    def __init__(self, model, include_hets=True, env_radius=10.0, device='cpu'):
        self.model = model
        self.include_hets = include_hets
        self.env_radius = env_radius
        self.device = device
    
    def __call__(self, elem):
        atom_df = elem['atoms']
        try:
            atom_df = first_model_filter(atom_df)
            atom_df = atom_df[~atom_df.hetero.str.contains('W')]
            atom_df = atom_df[atom_df.element != 'H'].reset_index(drop=True)
            if not self.include_hets:
                atom_df = atom_df[atom_df.resname.isin(atom_info.aa)].reset_index(drop=True)
        except:
            return None

        outdata = embed_protein(atom_df, self.model, device=self.device, include_hets=self.include_hets, env_radius=self.env_radius)
        if outdata is None:
            return
        elem['resids'] = outdata['resids']
        elem['confidence'] = outdata['confidence']
        elem['chains'] = outdata['chains']
        elem['embeddings'] = outdata['embeddings']
        return elem


class CDDGraphDataset(IterableDataset):
    '''
    Transforms LMDB dataset entries
    to featurized graphs. Returns a `torch_geometric.data.Data`
    graph
    '''
    
    def __init__(self, dataset, env_radius=10.0, device='cpu'):
        self.dataset = load_dataset(dataset, 'lmdb')
        self.env_radius = env_radius
        self.device = device
    
    def __iter__(self):
        for elem in self.dataset:
            cdd_id = elem['id']
            resids = elem['residue_ids']
            atoms = elem['atoms']
            atoms = first_model_filter(atoms)
            atoms = atoms[~atoms.hetero.str.contains('W')]
            atoms = atoms[atoms.element != 'H'].reset_index(drop=True)
            
            with open(f'msa_pdb_aligned/{cdd_id}.afa') as f:
                msa = MSA(AlignIO.read(f, 'fasta'))
            
            conservation = msa.get_conservation(msa.full_msa)
            dist_to_consensus = msa.get_dist_to_consensus()
            
            seq_pos_mat = msa.seq_pos_matrix()
            for c, colm in enumerate(seq_pos_mat.T):
                graphs = []
                all_resids = []
                pdb_ids = []
                rows = []
                for r, row in enumerate(colm):
                    if row == -1:
                        continue
                    try:
                        resid = resids[r][row]
                    except IndexError:
                        continue
                    pdb = elem['pdb_ids'][r]
                    chain = pdb[4]
                    df = atoms[(atoms['ensemble'] == pdb[:4] + '_' + pdb[4] + '.pdb') & (atoms['chain'] == chain)]
                    out = extract_env_from_resid(df.copy(), (chain, resid), self.env_radius)
                    pdb_ids.append(pdb)
                    graphs.append(out)
                    all_resids.append(resid)
                    rows.append(r)
                yield graphs, pdb_ids, all_resids, cdd_id, c, conservation[c], dist_to_consensus[rows, :][:, rows]
    
        
class MSPTransform(object):
    def __init__(self, env_radius=10.0, device=None):
        self.env_radius = env_radius
        self.device = device
    
    def __call__(self, item):
        mutation = item['id'].split('_')[-1]
        orig_df = item['original_atoms']
        mut_df = item['mutated_atoms']
        orig_df = orig_df[orig_df.element != 'H'].reset_index(drop=True)
        mut_df = mut_df[mut_df.element != 'H'].reset_index(drop=True)
    
        orig_center = self._extract_mut_coords(orig_df, mutation)
        mut_center = self._extract_mut_coords(mut_df, mutation)
        
        orig = extract_env_from_coords(orig_df, orig_center)
        mut = extract_env_from_coords(mut_df, mut_center)

        y = torch.FloatTensor([int(x) for x in item['label']])
        if (orig is None) or (mut is None):
            # print('skipping')
            return None 
        return orig, mut, y
    
    def _extract_mut_coords(self, df, mutation):
        chain, res = mutation[1], int(mutation[2:-1])
        idx = df.index[(df.chain.values == chain) & (df.residue.values == res)].values
        res_df = df.loc[idx, :]
        coords = sample_functional_center(res_df, mutation[0])
        # mask = torch.zeros(len(df), dtype=torch.long, device=self.device)
        # mask[idx] = 1
        return coords

class PPIDataset(IterableDataset):
    '''
    A `torch.utils.data.IterableDataset` wrapper around a
    ATOM3D PPI dataset.
    
    Modified from
    https://github.com/drorlab/atom3d/blob/master/examples/ppi/gnn/data.py
    and
    https://github.com/drorlab/gvp-pytorch/blob/main/gvp/atom3d.py

    '''
    def __init__(self, lmdb_dataset, env_radius=10.0, db5=False):
        self.dataset = load_dataset(lmdb_dataset, 'lmdb')
        self.env_radius = env_radius
        self.db5 = db5
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            gen = self._dataset_generator(list(range(len(self.dataset))), shuffle=True)
        else:  
            per_worker = int(math.ceil(len(self.dataset) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.dataset))
            gen = self._dataset_generator(
                list(range(len(self.dataset)))[iter_start:iter_end],
                shuffle=True)
        return gen

    def _extract_env(self, df, chain_res, label):
        df = df[df.element != 'H'].reset_index(drop=True)
        df = df[~df.hetero.str.contains('W')].reset_index(drop=True)
        chain, resnum = chain_res
        rows = (df['chain'] == chain) & (df['residue'] == int(resnum))
        res_df = df[rows]
        if 'CA' not in res_df.name.tolist():
            return None
        resname = res_df.resname.tolist()[0]
        resid = atom_info.aa_to_letter(resname) + str(resnum)
        if resid[0] == 'X':
            return None
        center = sample_functional_center(res_df, resid)
            
        df = df.reset_index()
        kd_tree = scipy.spatial.cKDTree(df[['x', 'y', 'z']].to_numpy())
        distances, idx = kd_tree.query(center, k=250)
        within_r = distances <= self.env_radius
        pt_idx = idx[within_r]

        df_env = df.iloc[pt_idx, :]
        
        if len(df_env) == 0:
            return None
        graph = transform(df_env)
        graph.y = label
        
        return graph

    def _dataset_generator(self, indices, shuffle=True):
        if shuffle:
            random.shuffle(indices)
        with torch.no_grad():
            for idx in indices:
                data = self.dataset[idx]

                neighbors = data['atoms_neighbors']
                pairs = data['atoms_pairs']
                
                for i, (ensemble_name, target_df) in enumerate(pairs.groupby(['ensemble'])):
                    sub_names, (bound1, bound2, unbound1, unbound2) = nb.get_subunits(target_df)
                    positives = neighbors[neighbors.ensemble0 == ensemble_name]
                    if self.db5:
                        negatives = nb.get_negatives(positives, unbound1, unbound2)
                    else:
                        negatives = nb.get_negatives(positives, bound1, bound2)
                    negatives['label'] = 0
                    labels = self._create_labels(positives, negatives, num_pos=10, neg_pos_ratio=1)
                    
                    for index, row in labels.iterrows():
                    
                        label = float(row['label'])
                        chain_res1 = row[['chain0', 'residue0']].values
                        chain_res2 = row[['chain1', 'residue1']].values
                        try:
                            graph1 = self._extract_env(bound1, chain_res1, label)
                            graph2 = self._extract_env(bound2, chain_res2, label)
                        except KeyError:
                            continue
                        if (graph1 is None) or (graph2 is None):
                            continue
                        yield graph1, graph2

    def _create_labels(self, positives, negatives, num_pos, neg_pos_ratio):
        frac = min(1, num_pos / positives.shape[0])
        positives = positives.sample(frac=frac)
        n = positives.shape[0] * neg_pos_ratio
        n = min(negatives.shape[0], n)
        negatives = negatives.sample(n, random_state=0, axis=0)
        labels = pd.concat([positives, negatives])[['chain0', 'residue0', 'chain1', 'residue1', 'label']]
        return labels


class PDBDataset(IterableDataset):
    '''
    Yields graphs for each residue environment of a PDB dataset containing PDB chains.
    '''
    
    def __init__(self, dataset, subset_idx=None, env_radius=10.0, device='cpu'):
        self.dataset = load_dataset(dataset, 'pdb')
        if subset_idx is not None:
            self.dataset = torch.utils.data.Subset(self.dataset, subset_idx)
        self.env_radius = env_radius
        self.device = device
    
    def __iter__(self):
        for elem in self.dataset:
            atoms = elem['atoms']
            try:
                atoms = first_model_filter(atoms)
                atoms = atoms[~atoms.hetero.str.contains('W')]
                atoms = atoms[atoms.element != 'H'].reset_index(drop=True)
            except:
                continue
            for (c, r, i), _ in atoms.groupby(['chain', 'resname', 'residue']):
                if r not in atom_info.aa:
                    continue
                chain_res = c + '_' + str(i)
                resid = atom_info.aa_to_letter(r) + str(i)
                try:
                    out = extract_env_from_resid(atoms.copy(), (c, resid), self.env_radius)
                except:
                    continue
                if out is None:
                    continue
                yield out, chain_res, resid, elem['id'].split('.')[0]
            

class PDBSampleDataset(IterableDataset):
    '''
    Yields graphs for each residue environment of a PDB dataset containing PDB chains.
    '''
    
    def __init__(self, dataset, train_mode=False, subset_idx=None, n_samples=5, env_radius=10.0, device='cpu'):
        self.dataset = load_dataset(dataset, 'pdb')
        if subset_idx is not None:
            self.dataset = torch.utils.data.Subset(self.dataset, subset_idx)
        self.env_radius = env_radius
        self.n_samples = n_samples
        self.device = device
        self.train_mode = train_mode
    
    def __iter__(self):
        for elem in self.dataset:
            atoms = elem['atoms']
            try:
                atoms = first_model_filter(atoms)
                atoms = atoms[~atoms.hetero.str.contains('W')]
                atoms = atoms[atoms.element != 'H'].reset_index(drop=True)
            except:
                continue
            for (c, r), rdf in atoms.groupby(['chain', 'resname']):
                if r not in atom_info.aa[:20]:
                    continue
                sampled_resids = np.random.choice(rdf.residue.unique(), size=self.n_samples)
                for resi in sampled_resids:
                    resid = atom_info.aa_to_letter(r) + str(resi)
                    out = extract_env_from_resid(atoms.copy(), (c, resid), self.env_radius, train_mode=self.train_mode)
                    if out is None:
                        continue
                    yield out, r
    
    
class SiteCoordDataset(IterableDataset):
    '''
    Yields graphs from a dictionary of xyz coordinates defining PDB sites.
    '''
    
    def __init__(self, dataset, pdb_dir, env_radius=10.0, device='cpu'):
        self.dataset = dataset
        self.pdb_dir = pdb_dir
        self.env_radius = env_radius
        self.device = device
    
    def __iter__(self):
        for it, (pdb, samples) in enumerate(self.dataset.items()):
            fp = os.path.join(self.pdb_dir, pdb + '.pdb.gz')
            if not os.path.exists(fp):
                print('skipping PDB', pdb)
                continue
            atoms = process_pdb(fp)

            for i, xyz in enumerate(samples['coords']):
                graph = extract_env_from_coords(atoms.copy(), xyz)
                if graph is None:
                    continue
                label = samples['labels'][i]
                yield graph, pdb, label
    
    def __len__(self):
        total = 0
        for pdb, samples in self.dataset.items():
            total += len(samples['labels'])
        return total
    
class SiteDataset(IterableDataset):
    '''
    Yields graphs from a dictionary of xyz coordinates defining PDB sites.
    '''
    
    def __init__(self, dataset, pdb_dir, train_mode=True, env_radius=10.0, device='cpu'):
        self.dataset = pd.read_csv(dataset, converters={'locs': lambda x: eval(x)})
        self.pdb_dir = pdb_dir
        self.env_radius = env_radius
        self.device = device
        self.train_mode = train_mode
    
    def __iter__(self):
        for pdb_chain, df in self.dataset.groupby('pdb'):
            pdb = pdb_chain[:4]
            chain = pdb_chain[4:]
            fp = os.path.join(self.pdb_dir, pdb + '.pdb.gz')
            if not os.path.exists(fp):
                print('skipping PDB', pdb)
                continue
            atom_df = process_pdb(fp, chain=chain)

            for r, (site, _, locs, source, desc) in df.iterrows():
                for resnum in locs:
                    if resnum not in atom_df.residue.tolist():
                        continue
                    try:
                        graph = self._extract_env(atom_df.copy(), resnum)
                    except KeyError:
                        print(pdb_chain, desc)
                        continue
                    if graph is None:
                        continue
                    yield graph, pdb_chain, source, desc

    def _extract_env(self, df, resnum):
        df = df.reset_index()
        kd_tree = scipy.spatial.cKDTree(df[['x', 'y', 'z']].to_numpy())
        rows = df['residue'] == int(resnum)
        res_df = df[rows]
        
        resid = res_df['resname'].apply(atom_info.aa_to_letter).iloc[0] + str(resnum)
        center = sample_functional_center(res_df, resid, self.train_mode)

        pt_idx = kd_tree.query_ball_point(center, r=self.env_radius, p=2.0)
        df_env = df.iloc[pt_idx, :]
        
        if len(df_env) == 0:
            return None
        graph = self.transform(df_env)
        graph.resid = resid
        
        return graph
    
    def __len__(self):
        total = 0
        for r, row in self.dataset.iterrows():
            total += len(row[2])
        return total
    
    
class MSA(MultipleSeqAlignment):
    def __init__(self, alignment, name=None, pdb_filter=None):
        self.pdb_filter = pdb_filter
        self.name = name
        self.full_alignment_length = alignment.get_alignment_length()
        self.full_msa = alignment
        self.consensus = alignment[0].seq
        # self.conservation = self.get_conservation(alignment)
        self.pdb_seq_records = list(filter(self._pdb_seq_filter, alignment))
        
        pdb_records = list(filter(self._pdb_filter, alignment))
        MultipleSeqAlignment.__init__(self, pdb_records)
        
    def get_dist_to_consensus(self):
        from Levenshtein import distance as lev
        n = len(self.pdb_seq_records)
        dists = np.zeros((n, n))
        for i, r1 in enumerate(self.pdb_seq_records):
            for j, r2 in enumerate(self.pdb_seq_records):
                indices = [i for i in range(len(r1.seq)) if str(r1.seq)[i] + str(r2.seq)[i] != '--']
                r1seq = ''.join([r1.seq[i] for i in indices])
                r2seq = ''.join([r2.seq[i] for i in indices])
                dist = lev(r1seq, r2seq) / float(len(r1seq))
                dists[i, j] = dist
        return dists
        
    def seq_pos_matrix(self):
        mat = np.zeros((len(self.pdb_seq_records), self.get_alignment_length()))
        for seq, record in enumerate(self.pdb_seq_records):
            gap_ct = 0
            for pos, aa in enumerate(record):
                if aa == '-':
                    gap_ct += 1
                    mat[seq, pos] = -1
                else:
                    mat[seq, pos] = pos - gap_ct
        return mat.astype(int)
    
    def get_domain_boundaries(self):
        records = self._records
        start = None
        end = None
        for i in range(self.get_alignment_length()):
            if start is not None:
                break
            for r in range(len(records)):
                val = records[r][i]
                if val != '-':
                    start = i
                    break
        for i in reversed(range(self.get_alignment_length())):
            if end is not None:
                break
            for r in range(len(records)):
                val = records[r][i]
                if val != '-':
                    end = i
                    break
        return start, end
    
    def _pdb_filter(self, record):
        if ('|pdb|' in record.id) or ('|sp|' in record.id):
            return True
        return False
    
    def _pdb_seq_filter(self, record):
        if '|' not in record.id:
            return True
        return False

    def get_pdb_chains(self):
        labels = []
        for r in self._records:
            split = r.id.split('|')
            if '|pdb|' in r.id:
                if split[3].strip() == '':
                    continue
                elif (len(split) < 5) or (split[4].strip() == ''):
                    labels.append(split[3].lower() + '_A')
                else:
                    labels.append(split[3].lower() + '_' + split[4])
            elif '|sp|' in r.id:
                if split[3].strip() == '':
                    continue
                labels.append(r.id.split('|')[3] + '_A')
        return labels
    
    def get_aligned_positions(self):
        gap_cols = np.array([[col == '-' for col in row] for row in self._records])
        valid_positions = np.sum(gap_cols, axis=0) == 0
        return np.nonzero(valid_positions)[0]
    
    def get_aligned_positions_pairwise(self, r1, r2, seq_r1, seq_r2):
        valid_positions = [i for i in range(self.get_alignment_length()) if '-' not in r1[i] + r2[i] + seq_r1[i] + seq_r2[i]]
        return valid_positions
    
    def sequence_identity(self, r1, r2):
        matches = [r1[i] == r2[i] for i in range(self.get_alignment_length()) if r1[i] + r2[i] != '--']
        return np.mean(matches)
    
    def sample_record_pair(self):
        seq_i1, seq_i2 = np.random.choice(range(len(self.pdb_seq_records)), size=2, replace=False)
        seq_r1, seq_r2 = self.pdb_seq_records[seq_i1], self.pdb_seq_records[seq_i2]
        if len(seq_r1.id.split('_')[0]) == 4:
            r1 = [r for r in self._records if seq_r1.id.split('_')[0].upper() + '|' + seq_r1.id.split('_')[1] in r.id][0]
        else:
            r1 = [r for r in self._records if seq_r1.id.split('_')[0].upper() in r.id][0]
        if len(seq_r2.id.split('_')[0]) == 4:
            r2 = [r for r in self._records if seq_r2.id.split('_')[0].upper() + '|' + seq_r2.id.split('_')[1] in r.id][0]
        else:
            r2 = [r for r in self._records if seq_r2.id.split('_')[0].upper() in r.id][0]
        return r1, r2, seq_r1, seq_r2

    def sample_position_pairs(self, r1, r2, seq_r1, seq_r2, num_pairs=1):
        sample_pos = np.random.choice(self.get_aligned_positions_pairwise(r1, r2, seq_r1, seq_r2), size=num_pairs, replace=False)
        
        if num_pairs == 1:
            sample_pos = int(sample_pos[0])
            cons = 1 / (self.calculate_entropy(self.full_msa[:, sample_pos]) + 1)
            pos1 = self.align_pos_to_seq_pos(sample_pos, seq_r1)
            pos2 = self.align_pos_to_seq_pos(sample_pos, seq_r2)
            return pos1, pos2, cons
        else:
            pos_pairs = []
            for pos in sample_pos:
                pos1 = self.align_pos_to_seq_pos(pos, seq_r1)
                pos2 = self.align_pos_to_seq_pos(pos, seq_r2)
                pos_pairs.append((pos1, pos2))
            return pos_pairs
        
    def align_pos_to_seq_pos(self, pos, record):
        gap_ct = 0
        for i, aa in enumerate(record):
            if i == pos:
                break
            if aa == '-':
                gap_ct += 1
        return pos - gap_ct
  
    def calculate_entropy(self, column):
        aas = col.Counter([aa for aa in column if aa != '-'])
        counts = aas.values()
        dist = [x / sum(counts) for x in counts]
        entropy_value = entropy(dist, base=2)

        return entropy_value
    
    def get_conservation(self, msa):
        return 1 / (np.array([self.calculate_entropy(msa[:, c]) for c in range(self.full_alignment_length)]) + 1)


class NoneCollater:
    def __init__(self, follow_batch=None, exclude_keys=None):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch, filter_batch=True):
        if filter_batch:
            batch = list(filter(lambda x: (x[0][0] is not None) & (x[0][1] is not None), batch))
        elem = batch[0]
        if isinstance(elem, BaseData):
            return Batch.from_data_list(batch, self.follow_batch,
                                        self.exclude_keys)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch], filter_batch=False) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s, filter_batch=False) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s, filter_batch=False) for s in zip(*batch)]

        raise TypeError(f'DataLoader found invalid type: {type(elem)}')

    def collate(self, batch):  # Deprecated...
        return self(batch)
