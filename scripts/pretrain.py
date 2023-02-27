import torch
import torch.nn.functional as F
import numpy as np
from collapse.byol_pytorch import BYOL
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from collapse.models import CDDModel
from collapse.data import CDDTransform, NoneCollater
from atom3d.datasets import load_dataset
from torch_geometric.nn import DataParallel
from torch.utils.data import DataLoader
from torch_geometric.loader import DataListLoader

import os
import argparse
import datetime
from tqdm import tqdm
import wandb

# torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description='BYOL implementation for aligned protein environments')
parser.add_argument('--val_dir', type=str, default='/scratch/users/aderry/collapse/datasets/pfam_val_dataset_msa',
                    help='location of dataset')
parser.add_argument('--data_dir', type=str, default='/scratch/users/aderry/collapse/datasets/cdd_train_dataset',
                    help='location of dataset')
parser.add_argument('--run_name', type=str, default=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), 
                    help='run name for logging')
parser.add_argument('--checkpoint', type=str, default="", 
                    help='load weights from checkpoint file')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--dim', default=512, type=int, 
                    help='dimensionality of learned representations')
parser.add_argument('--edge_cutoff', default=4.5, type=float, 
                    help='cutoff for defining spatial graph')
parser.add_argument('--env_radius', default=10.0, type=float, 
                    help='radius of atomic environment')
parser.add_argument('--parallel', action='store_true',
                    help='Use multiple GPUs')
parser.add_argument('--tied_weights', action='store_true',
                    help='Use tied weights for target and online encoder (as in SimSiam)')

args = parser.parse_args()

NUM_GPUS = torch.cuda.device_count()
print(f'using {NUM_GPUS} GPUs')


@torch.no_grad()
def evaluate(loader, model, device):
    model.eval()
    cls_x = []
    cls_y = []
    losses = []
    pos_cosine = []
    neg_cosine = []
    for i, ((graph1, graph2), meta) in enumerate(loader):
        graph1 = graph1.to(device)
        graph2 = graph2.to(device)
        res1, res2 = meta['res_labels']
        cons = meta['conservation'].to(device)
        loss = model(graph1, graph2, cons)
        losses.append(loss.item())
        
        # record std of embeddings (to check for collapsing solution)
        embeddings = model(graph1, graph2, return_embedding=True, return_projection=False)
        
        a, b = embeddings
        pos_cosine.extend(F.cosine_similarity(a, b).tolist())
        neg_cosine.extend(F.cosine_similarity(a, b[torch.randperm(b.size(0))]).tolist())
        embeddings = torch.cat(embeddings).cpu().detach()
        
        cls_x.append(embeddings)
        cls_y.append(torch.cat([res1, res2]))
    
    cls_x = torch.cat(cls_x)
    cls_y = torch.cat(cls_y)
    emb_norm = cls_x / torch.linalg.norm(cls_x, dim=1).unsqueeze(1)
    
    acc = train_cls(emb_norm, cls_y)
    
    std = torch.std(emb_norm, dim=0)
    return np.mean(losses), acc, std.mean(), np.mean(pos_cosine), np.mean(neg_cosine)

def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    wandb.init(project="collapse", name=args.run_name, config=vars(args))
    
    train_dataset = load_dataset(args.data_dir, 'lmdb', transform=CDDTransform(single_chain=True, include_af2=False, env_radius=args.env_radius, num_pairs_sampled=1))
    val_dataset = load_dataset(args.val_dir, 'lmdb', transform=CDDTransform(single_chain=True, include_af2=False, env_radius=args.env_radius, num_pairs_sampled=1))
    
    dummy_graph = torch.load(os.path.join(os.environ["DATA_DIR"], 'dummy_graph.pt'))
    
    model = BYOL(
        CDDModel(out_dim=args.dim, scatter_mean=True, attn=False, chain_ind=False),
        projection_size=512,
        dummy_graph=dummy_graph,
        hidden_layer=-1,
        use_momentum=(not args.tied_weights)
    ).to(device)
    
    device_ids = [i for i in range(torch.cuda.device_count())]

    opt = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000, eta_min=1e-6)

    if args.checkpoint != "":
        cpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(cpt['model_state_dict'])
        # scheduler.load_state_dict(cpt['scheduler_state_dict'])
        opt.load_state_dict(cpt['optimizer_state_dict'])
        args.start_epoch = cpt['epoch']
        
    if args.parallel:
        # print(f"Using {len(device_ids)} GPUs")
        model = DataParallel(model, device_ids=device_ids)
        train_loader = DataListLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, collate_fn=NoneCollater(), pin_memory=True, persistent_workers=True)
        val_loader = DataListLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, collate_fn=NoneCollater(), pin_memory=True, persistent_workers=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=NoneCollater(), pin_memory=True, persistent_workers=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=NoneCollater(), pin_memory=True, persistent_workers=True)
          
    model.train()
    
    # wandb.watch(model)
    
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        model.train()
        # print(f'EPOCH {epoch+1}:')

        for i, ((graph1, graph2), meta) in enumerate(train_loader):
            # if i == 2:
            #     break
            graph1 = graph1.to(device)
            graph2 = graph2.to(device)
            cons = meta['conservation'].to(device)
            with torch.cuda.amp.autocast():
                try:
                    loss = model(graph1, graph2, loss_weight=cons, return_projection=True)
                except RuntimeError as e:
                    if "CUDA out of memory" not in str(e): raise(e)
                    torch.cuda.empty_cache()
                    print('Out of Memory error!', flush=True)
                    continue
            
            # print(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if not args.tied_weights:
                model.update_moving_average(epoch) # update moving average of target encoder
            wandb.log({'loss': loss.item()})
            # print(f'Iteration {i}: Loss: {loss.item()}')
        
        val_loss, acc, std, pos_cosine, neg_cosine = evaluate(val_loader, model, device)
        wandb.log({'epoch': epoch, 'val_loss': val_loss, 'aa_knn_acc': acc, 'std': std, 'pos_cosine': pos_cosine, 'neg_cosine': neg_cosine})
        
        # save your improved network
        if args.parallel:
            torch.save({'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        # 'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch
                        }, f'../data/checkpoints/{args.run_name}.pt')
        else:
            torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    # 'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch
                    }, f'../data/checkpoints/{args.run_name}.pt')
        # scheduler.step(val_loss)

def train_cls(x, y):
    # mod = LogisticRegression(max_iter=100, solver="liblinear")
    mod = KNeighborsClassifier(5, metric='cosine')
    scores = cross_val_score(mod, x, y, cv=4)
    return np.mean(scores)

if __name__ == "__main__":
    main()
