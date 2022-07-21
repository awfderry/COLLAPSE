import numpy as np
import os
import argparse
import torch
from collapse.data import EmbedTransform
from atom3d.datasets import load_dataset, make_lmdb_dataset
import atom3d.util.file as fi
from collapse import initialize_model

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str)
parser.add_argument('out_dir', type=str)
parser.add_argument('--split_id', type=int, default=0)
parser.add_argument('--checkpoint', type=str, default='data/checkpoints/collapse_base.pt')
parser.add_argument('--filetype', type=str, default='pdb')
parser.add_argument('--num_splits', type=int, default=1)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = initialize_model(args.checkpoint, device=device)
transform = EmbedTransform(model, device=device)
dataset = load_dataset(args.data_dir, args.filetype, transform=transform)

if args.num_splits > 1:
    out_path = os.path.join(args.out_dir, f'tmp_{args.split_id}')
    os.makedirs(out_path, exist_ok=True)

    all_files = fi.find_files(args.data_dir, args.filetype)
    split_idx = np.array_split(np.arange(len(all_files)), args.num_splits)[args.split_id - 1]
    print(f'Processing split {args.split_id} with {len(split_idx)} examples...')

    dataset = torch.utils.data.Subset(dataset, split_idx)
else:
    out_path = args.out_dir
    print(f'Processing full dataset with {len(dataset)} examples...')

make_lmdb_dataset(dataset, out_path, serialization_format='pkl', filter_fn=lambda x: (x is None))

# e_db = []
# pdb_ids = []
# chains = []
# resids = []
# for elem in tqdm(dataset):
#     if elem is None:
#         continue
#     name, _ = pdb_from_fname(elem['id'])
#     resids.extend(elem['chains'])
#     resids.extend(elem['resids'])
#     pdb_ids.extend([name] * len(elem['resids']))
#     e_db.append(elem['embeddings'])

# e_db = np.stack(e_db, 0)

# print(e_db.shape)

# outdata = {'embeddings': e_db, 'pdbs': pdb_ids, 'chains': chains, 'resids': resids}

# with open(args.outfile, 'wb') as f:
#     pickle.dump(outdata, f)
