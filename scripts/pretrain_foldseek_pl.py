import torch
import torch.nn.functional as F
import numpy as np
from collapse.byol_pytorch import BYOL
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from collapse.models import CDDModel
from collapse.data import FoldseekDataset, NoneCollater
from atom3d.datasets import load_dataset
from torch_geometric.nn import DataParallel
from torch.utils.data import DataLoader
import torch.nn as nn
from torch_geometric.loader import DataLoader as PTGLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import os
import argparse
import datetime
from tqdm import tqdm
import wandb

# torch.autograd.set_detect_anomaly(True)

NUM_WORKERS = int(os.environ["SLURM_CPUS_PER_TASK"])
NUM_NODES = int(os.environ["SLURM_NNODES"])
ALLOCATED_GPUS_PER_NODE = int(os.environ["SLURM_GPUS_ON_NODE"])
print(f'using {ALLOCATED_GPUS_PER_NODE} GPUs')


def train_cls(x, y):
    # mod = LogisticRegression(max_iter=100, solver="liblinear")
    mod = KNeighborsClassifier(5, metric='cosine')
    try:
        scores = cross_val_score(mod, x, y, cv=4)
    except ValueError:
        return np.nan
    return np.mean(scores)


class CollapseModel(pl.LightningModule):
    def __init__(self, byol_model, **kwargs):
        super().__init__()
        self.byol_model = byol_model
        self.args = kwargs
        # self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        (graph1, graph2), meta = batch
        loss = self.byol_model(graph1, graph2, return_projection=True)
        self.logger.experiment.log({'loss': loss.item()})
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            (graph1, graph2), meta = batch
            res1, res2 = meta['res_labels']
            loss = self.byol_model(graph1, graph2, return_projection=True)
            a, _ = self.byol_model.online_encoder(graph1, return_projection=False)
            b, _ = self.byol_model.online_encoder(graph2, return_projection=False)
            pos_cosine = F.cosine_similarity(a, b)
            neg_cosine = F.cosine_similarity(a, b[torch.randperm(b.size(0))])
            cls_x = torch.cat([a, b])
            cls_y = torch.cat([res1, res2])
            
            return {'loss': loss, 'cls_x': cls_x, 'cls_y': cls_y, 'pos_cosine': pos_cosine, 'neg_cosine': neg_cosine}
            
        elif dataloader_idx == 1:
            g, label = batch
            embeddings, _ = self.byol_model.online_encoder(g, return_projection=False)
            # print(label)
            return {'embeddings': embeddings, 'labels': label}
    
    def validation_epoch_end(self, outputs):
        out_0, out_1 = outputs
        
        cls_x = torch.cat([x['cls_x'] for x in out_0])
        cls_y = torch.cat([x['cls_y'] for x in out_0])
        pos_cosine = torch.cat([x['pos_cosine'] for x in out_0]).mean()
        neg_cosine = torch.cat([x['neg_cosine'] for x in out_0]).mean()
        val_loss = torch.tensor([x['loss'] for x in out_0]).mean()
        
        emb_norm = (cls_x / torch.linalg.norm(cls_x, dim=1).unsqueeze(1))
        acc = train_cls(emb_norm.cpu().detach(), cls_y.cpu().detach())
        std = torch.std(emb_norm, dim=0).mean()
        
        self.logger.experiment.log({'epoch': self.current_epoch, 'val_loss': val_loss, 'aa_knn_acc': acc, 'std': std, 'pos_cosine': pos_cosine, 'neg_cosine': neg_cosine})
        
        cls_x = torch.cat([x['embeddings'] for x in out_1])
        cls_y = torch.cat([x['labels'] for x in out_1])
        # print(cls_x.shape, cls_y.shape)
        acc = train_cls(cls_x.cpu().detach(), cls_y.cpu().detach())
        self.logger.experiment.log({'prosite_acc': acc})

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['epochs'], eta_min=1e-6)
        return  {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
        
    def on_before_zero_grad(self, _):
        if self.byol_model.use_momentum:
            self.byol_model.update_moving_average()


def main(args):
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # wandb.init(project="collapse", name=args.run_name, config=vars(args))
    wandb_logger = WandbLogger(project="collapse", log_model="all", name=args.run_name)
    pl.seed_everything(77, workers=True)
    
    train_dataset = FoldseekDataset(args.train_dataset, '/scratch/users/aderry/foldseek/pdb.lookup', '/scratch/users/aderry/pdb_lmdb', num_positions=1)
    val_dataset = FoldseekDataset(args.val_dataset, '/scratch/users/aderry/foldseek/pdb.lookup', '/scratch/users/aderry/pdb_lmdb', num_positions=1)
    
    dummy_graph = torch.load(os.path.join(os.environ["DATA_DIR"], 'dummy_graph.pt'))
    
    byol_model = BYOL(
        CDDModel(out_dim=args.dim, scatter_mean=True, attn=False, chain_ind=False),
        projection_size=1048,
        dummy_graph=dummy_graph,
        hidden_layer=-1,
        use_momentum=(not args.tied_weights)
    )
    
    model = CollapseModel(byol_model, **vars(args))
    wandb_logger.watch(model)
    
    # device_ids = [i for i in range(NUM_GPUS)]

    # opt = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000, eta_min=1e-6)
    
    if args.checkpoint != "":
        model.load_from_checkpoint(args.checkpoint)
    #     cpt = torch.load(args.checkpoint, map_location=device)
    #     model.load_state_dict(cpt['model_state_dict'], strict=False)
    #     scheduler.load_state_dict(cpt['scheduler_state_dict'])
    #     # opt.load_state_dict(cpt['optimizer_state_dict'])
    #     args.start_epoch = cpt['epoch']
    
    # if args.parallel:
    #     print(f"Using {len(device_ids)} GPUs")
    #     model = DataParallel(model, device_ids=device_ids)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS, collate_fn=NoneCollater(), pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS, collate_fn=NoneCollater(), pin_memory=True, persistent_workers=True)
    
    prosite_dataset = load_dataset('../data/datasets/prosite_val_subset', 'lmdb', transform=lambda x: (x['graph'], x['label']))
    prosite_loader = PTGLoader(prosite_dataset, batch_size=16, shuffle=False)
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="epoch",
        mode="max",
        dirpath="../data/checkpoints",
        filename=f"{args.run_name}-" + "{epoch:02d}")
          
    trainer = pl.Trainer(max_epochs=args.epochs, default_root_dir='../data/checkpoints', num_nodes=NUM_NODES, devices=ALLOCATED_GPUS_PER_NODE, accelerator="gpu", strategy="ddp", logger=wandb_logger, precision=16, callbacks=[checkpoint_callback], sync_batchnorm=True)
    trainer.fit(model, train_loader, [val_loader, prosite_loader])
    trainer.save_checkpoint(f"../data/checkpoints/{args.run_name}-final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BYOL implementation for aligned protein environments')
    parser.add_argument('--val_dataset', type=str, default='/scratch/users/aderry/collapse/datasets/foldseek_msa_val',
                        help='location of dataset')
    parser.add_argument('--train_dataset', type=str, default='/scratch/users/aderry/collapse/datasets/foldseek_msa_train',
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
    parser.add_argument('--gpus', default=ALLOCATED_GPUS_PER_NODE, type=int,
                        help='number of GPUs for distributed training')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--weight_decay', default=0.001, type=float, help='weight decay')
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

    main(args)
