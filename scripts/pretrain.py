import torch
import torch.nn.functional as F
import numpy as np
from collapse.byol_pytorch import BYOL
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from collapse.models import CDDModel
from collapse.data import CDDTransform, NoneCollater
from atom3d.datasets import load_dataset
from torch_geometric.data import Batch
from torch.utils.data import DataLoader

import argparse
import datetime
from tqdm import tqdm
import wandb

parser = argparse.ArgumentParser(description='BYOL implementation for aligned protein environments')
parser.add_argument('--val_dir', type=str, default='../data/lmdb/pfam_pdb_balanced',
                    help='location of dataset')
parser.add_argument('--data_dir', type=str, default='../data/lmdb/cdd_pdb_dataset_new',
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
parser.add_argument('--lamb', default=0.0, type=float, 
                    help='lambda for weigting node pooling')
parser.add_argument('--tied_weights', action='store_true',
                    help='Use tied weights for target and online encoder (as in SimSiam)')

args = parser.parse_args()

NUM_GPUS = torch.cuda.device_count()
print(f'using {NUM_GPUS} GPUs')

    
def collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = list(filter(lambda x: x is not None, batch))
    batch = Batch.from_data_list(batch)
    return batch

@torch.no_grad()
def evaluate(loader, learner, device):
    learner.eval()
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
        loss = learner(graph1, graph2, cons)
        losses.append(loss.item())
        
        # record std of embeddings (to check for collapsing solution)
        embeddings = learner(graph1, graph2, return_embedding=True, return_projection=False)
        
        a, b = embeddings
        pos_cosine.extend(F.cosine_similarity(a, b).tolist())
        neg_cosine.extend(F.cosine_similarity(a, b[torch.randperm(b.size(0))]).tolist())
        embeddings = torch.cat(embeddings).cpu().detach()
        
        cls_x.append(embeddings)
        cls_y.append(torch.cat([res1, res2]))
    
    cls_x = torch.cat(cls_x)
    cls_y = torch.cat(cls_y)
    
    acc = train_cls(cls_x, cls_y)
    
    emb_norm = cls_x / torch.linalg.norm(cls_x, dim=1).unsqueeze(1)
    std = torch.std(emb_norm, dim=0)
    return np.mean(losses), acc, std.mean(), np.mean(pos_cosine), np.mean(neg_cosine)

def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # wandb_logger = WandbLogger(project="protein-ssl", log_model=True, name=args.run_name, config=vars(args))
    wandb.init(project="protein-ssl", name=args.run_name, config=vars(args))
    
    train_dataset = load_dataset(args.data_dir, 'lmdb', transform=CDDTransform(lamb=args.lamb, single_chain=True, env_radius=args.env_radius))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=NoneCollater())
    val_dataset = load_dataset(args.val_dir, 'lmdb', transform=CDDTransform(lamb=args.lamb, single_chain=True, env_radius=args.env_radius))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=NoneCollater())
    
    for (graph1, graph2), _ in train_loader:
        dummy_graph = graph1.clone()
        dummy_graph.x = torch.randn_like(dummy_graph.x)
        dummy_graph.edge_s = torch.randn_like(dummy_graph.edge_s)
        dummy_graph.edge_v = torch.randn_like(dummy_graph.edge_v)
        break
    
    learner = BYOL(
        CDDModel(out_dim=args.dim, scatter_mean=True, attn=False, chain_ind=False),
        projection_size=512,
        dummy_graph=dummy_graph,
        hidden_layer = -1,
        use_momentum = (not args.tied_weights)
    ).to(device)

    opt = torch.optim.Adam(params=learner.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    if args.checkpoint != "":
        cpt = torch.load(args.checkpoint, map_location=device)
        learner.load_state_dict(cpt['model_state_dict'])
        # scheduler.load_state_dict(cpt['scheduler_state_dict'])
        opt.load_state_dict(cpt['optimizer_state_dict'])
        args.start_epoch = cpt['epoch']
    learner.train()
    
    wandb.watch(learner)
    
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        learner.train()
        # print(f'EPOCH {epoch+1}:')

        for i, ((graph1, graph2), meta) in enumerate(train_loader):
            graph1 = graph1.to(device)
            graph2 = graph2.to(device)
            cons = meta['conservation'].to(device)
            loss = learner(graph1, graph2, loss_weight=cons, return_projection=True)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if not args.tied_weights:
                learner.update_moving_average() # update moving average of target encoder
            wandb.log({'loss': loss.item()})
            # print(f'Iteration {i}: Loss: {loss.item()}')
        
        val_loss, acc, std, pos_cosine, neg_cosine = evaluate(val_loader, learner, device)
        wandb.log({'epoch': epoch, 'val_loss': val_loss, 'aa_knn_acc': acc, 'std': std, 'pos_cosine': pos_cosine, 'neg_cosine': neg_cosine})
        
        # save your improved network
        torch.save({'model_state_dict': learner.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    # 'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch
                    }, f'../data/checkpoints/{args.run_name}.pt')
        # scheduler.step()

def train_cls(x, y):
    knn = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn, x, y, cv=4)
    return np.mean(scores)

if __name__ == "__main__":
    main()
