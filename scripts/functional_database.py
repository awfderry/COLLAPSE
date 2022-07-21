import numpy as np
import argparse
import pickle
from tqdm import tqdm
import torch
from collapse.data import SiteDataset
from torch_geometric.loader import DataLoader
from collapse import initialize_model

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('outfile', type=str)
parser.add_argument('--checkpoint', type=str, default='data/checkpoints/collapse_base.pt')
parser.add_argument('--pdb_dir', type=str, default='../data/pdb/prosite_pdb')
args = parser.parse_args()

# os.makedirs(args.outfile, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = SiteDataset(args.dataset, args.pdb_dir, train_mode=False)
loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

for g in loader:
    dummy_graph = g[0].clone()
    dummy_graph.x = torch.randn_like(dummy_graph.x)
    dummy_graph.edge_s = torch.randn_like(dummy_graph.edge_s)
    dummy_graph.edge_v = torch.randn_like(dummy_graph.edge_v)
    break

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
outdata = {'embeddings': all_emb, 'pdbs': all_pdb, 'resids': all_resids, 'sites': all_sites, 'sources': all_sources}
with open(args.outfile, 'wb') as f:
    pickle.dump(outdata, f)
