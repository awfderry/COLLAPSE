import argparse
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from collapse.byol_pytorch import BYOL
from collapse.models import CDDModel, MLPPaired
from collapse.data import PPIDataset

from sklearn.metrics import roc_auc_score, average_precision_score


@torch.no_grad()
def test(gnn_model, ff_model, loader, criterion, device):
    gnn_model.eval()
    ff_model.eval()

    losses = []

    y_true = []
    y_pred = []

    for it, (original, mutated) in enumerate(loader):
        original = original.to(device)
        mutated = mutated.to(device)
        out_original, _ = gnn_model.online_encoder(original, return_projection=False)
        out_mutated, _ = gnn_model.online_encoder(mutated, return_projection=False)
        output = ff_model(out_original, out_mutated)
        
        loss = criterion(output, original.y)
        losses.append(loss.item())
        y_true.extend(original.y.tolist())
        y_pred.extend(torch.sigmoid(output).tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)

    return np.mean(losses), auroc, auprc, y_true, y_pred


def evaluate(args, device, log_dir, rep=None):
    test_dataset = PPIDataset(args.data_dir, args.radius, db5=True)
    
    test_loader = DataLoader(test_dataset, args.batch_size, num_workers=4)
    
    for original, mutated in test_loader:
        dummy_graph = original.clone()
        dummy_graph.x = torch.randn_like(dummy_graph.x)
        dummy_graph.edge_s = torch.randn_like(dummy_graph.edge_s)
        dummy_graph.edge_v = torch.randn_like(dummy_graph.edge_v)
        break

    gnn_model = BYOL(
        CDDModel(out_dim=512, scatter_mean=True, attn=False),
        projection_size=512,
        dummy_graph=dummy_graph,
        hidden_layer = -1,
        use_momentum = True,
        dense=False
    ).to(device)
    ff_model = MLPPaired(args.hidden_dim).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    criterion.to(device)

    train_file = os.path.join(log_dir, f'ppi-rep{rep}.best.train.pt')
    val_file = os.path.join(log_dir, f'ppi-rep{rep}.best.val.pt')
    shutil.copy(train_file, train_file.replace('ppi-rep', 'db5-rep'))
    shutil.copy(val_file, val_file.replace('ppi-rep', 'db5-rep'))
    test_file = os.path.join(log_dir, f'db5-rep{rep}.best.test.pt')
    cpt = torch.load(os.path.join(args.checkpoint, f'best_weights_rep{rep}.pt'), map_location=device)
    gnn_model.load_state_dict(cpt['gcn_state_dict'])
    ff_model.load_state_dict(cpt['ff_state_dict'])
    test_loss, auroc, auprc, y_true_test, y_pred_test = test(gnn_model, ff_model, test_loader, criterion, device)
    print(f'\tTest loss {test_loss}, Test AUROC {auroc}, Test auprc {auprc}')
    torch.save({'targets': y_true_test, 'predictions': y_pred_test}, test_file)
    return test_loss, auroc, auprc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/scratch/users/raphtown/atom3d_mirror/lmdb/PPI/raw/DB5/data')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--radius', type=float, default=10.0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = os.path.join('logs', f'ppi_{args.checkpoint}')
    
    args.checkpoint = os.path.join('logs', f'ppi_{args.checkpoint}')
    
    for rep, seed in enumerate(np.random.randint(0, 1000, size=3)):
        print('seed:', seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        evaluate(args, device, log_dir, rep + 1)
