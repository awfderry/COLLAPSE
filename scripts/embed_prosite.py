import numpy as np
import os
import json
import argparse
import pickle
from tqdm import tqdm
import torch
from collapse.data import SiteCoordDataset
from torch_geometric.loader import DataLoader
from collapse import initialize_model

parser = argparse.ArgumentParser()
parser.add_argument('site_name', type=str)
parser.add_argument('out_dir', type=str)
parser.add_argument('--checkpoint', type=str, default='../data/checkpoints/collapse_base.pt')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pdb_dir = '/oak/stanford/groups/rbaltman/aderry/COLLAPSE/data/prosite_pdb'

print(args.site_name)

with open(f'../data/prosite_data_{args.site_name}.json') as f:
    dataset = json.load(f)

dataset = SiteCoordDataset(dataset, pdb_dir, env_radius=10.0)
loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

model = initialize_model(args.checkpoint, device=device, train=False, attn=False)

loader = DataLoader(dataset, batch_size=1, shuffle=False)


print('Computing TP/TN embeddings...')
all_emb = []
prosite_labels = []
all_pdb = []
with torch.no_grad():
    for g, pdb, label in tqdm(loader, desc='TP/TN'):
        g = g.to(device)
        embeddings, _ = model.online_encoder(g, return_projection=False)
        all_emb.append(embeddings.squeeze().cpu().numpy())
        all_pdb.append(pdb[0])
        if label[0] == 0:
            prosite_labels.append('TN')
        elif label[0] == 1:
            prosite_labels.append('TP')

test_file = f'../data/prosite_test_{args.site_name}.json'
if os.path.exists(test_file):
    with open(test_file) as f:
        dataset = json.load(f)
    dataset = SiteCoordDataset(dataset, pdb_dir, env_radius=10.0)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print('Computing FP/FN embeddings...')
    with torch.no_grad():
        for g, pdb, label in tqdm(loader, desc='FP/FN'):
            g = g.to(device)
            embeddings, _ = model.online_encoder(g, return_projection=False)
            all_emb.append(embeddings.squeeze().cpu().numpy())
            all_pdb.append(pdb[0])
            if label[0] == 1:
                prosite_labels.append('FN')
            elif label[0] == 0:
                prosite_labels.append('FP')
     
print('Saving...')   
all_emb = np.stack(all_emb)
outdata = {'embeddings': all_emb, 'pdbs': all_pdb, 'labels': prosite_labels}
outfile = os.path.join(args.out_dir, f'{args.site_name}.pkl')
with open(outfile, 'wb') as f:
    pickle.dump(outdata, f)
