import numpy as np
import os
import argparse
from fastdist import fastdist
import pickle
from tqdm import tqdm
import torch
from collapse.data import SiteDataset
from torch_geometric.loader import DataLoader
from collapse import initialize_model, atom_info

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('outfile', type=str)
parser.add_argument('--checkpoint', type=str, default='data/checkpoints/collapse_base.pt')
parser.add_argument('--pdb_dir', type=str, default='../data/prosite_pdb')
args = parser.parse_args()

# os.makedirs(args.outfile, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = SiteDataset(args.dataset, args.pdb_dir, train_mode=False)
loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

model = initialize_model(args.checkpoint, device=device)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

print('Computing embeddings...')
all_emb = []
prosite_labels = []
all_pdb = []
all_sites = []
all_sources = []
all_resids = []
with torch.no_grad():
    for g, pdb, source, desc in tqdm(loader):
        g = g.to(device)
        embeddings, _ = model.online_encoder(g, return_projection=False)
        all_emb.append(embeddings.squeeze().cpu().numpy())
        all_pdb.append(pdb[0])
        all_sites.append(desc[0])
        all_sources.append(source[0])
        all_resids.append(g.resid[0])
     
print('Saving...')
all_emb = np.stack(all_emb)
outdata = {'embeddings': all_emb.copy(), 'pdbs': all_pdb, 'resids': all_resids, 'sites': all_sites, 'sources': all_sources}
# with open(args.outfile, 'wb') as f:
#     pickle.dump(outdata, f)

restype_to_emb = {}
for resname in atom_info.aa[:20]:
    restype_to_emb[resname] = np.load(os.path.join('/scratch/users/aderry/background_embeddings', f'{resname}.npy'))

pvals = [1e-2, 9e-3, 8e-3, 7e-3, 6e-3, 5e-3, 4e-3, 3e-3, 2e-3, 1e-3, 9e-4, 8e-4, 7e-4, 6e-4, 5e-4, 4e-4, 3e-4, 2e-4, 1e-4, 9e-5, 8e-5, 7e-5, 6e-5, 5e-5, 4e-5, 3e-5, 2e-5, 1e-5]
quants = [1-p for p in pvals]

mean_cos = []
std_cos = []
site_quants = []
for i in tqdm(range(len(all_emb))):
    emb = all_emb[i]
    restype = atom_info.letter_to_aa(all_resids[i][0])
    if restype == 'UNK':
        continue
    ref_emb = restype_to_emb[restype]
    # print(emb.shape)
    # print(ref_emb.shape)
    cosines = fastdist.cosine_matrix_to_matrix(emb[np.newaxis, :], ref_emb)[0]
    
    quantiles = np.quantile(cosines, quants)
    # all_cos.extend(cos_list)
    data = {pvals[j]:quantiles[j] for j in range(len(quantiles))}
    mean_cos.append(np.mean(cosines))
    std_cos.append(np.std(cosines))
    site_quants.append(data)
    
outdata['mean_cos'] = np.array(mean_cos)
outdata['std_cos'] = np.array(std_cos)
outdata['quantiles'] = site_quants
with open(args.outfile, 'wb') as f:
    pickle.dump(outdata, f)