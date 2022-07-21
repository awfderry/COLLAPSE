import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from byol_pytorch import BYOL
from models import CDDModel, MLPPaired
from data import PPIDataset

from sklearn.metrics import roc_auc_score, average_precision_score

def train_loop(epoch, gnn_model, ff_model, loader, criterion, optimizer, device):
    gnn_model.train()
    ff_model.train()

    start = time.time()

    losses = []
    print_frequency = 1000
    for it, (original, mutated) in enumerate(loader):
        original = original.to(device)
        mutated = mutated.to(device)
        optimizer.zero_grad()
        out_original, _ = gnn_model.online_encoder(original, return_projection=False)
        out_mutated, _ = gnn_model.online_encoder(mutated, return_projection=False)
        output = ff_model(out_original, out_mutated)
        loss = criterion(output, original.y)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

        if it % print_frequency == 0:
            elapsed = time.time() - start
            print(f'Epoch {epoch}, iter {it}, train loss {np.mean(losses)}, avg it/sec {print_frequency / elapsed}')
            start = time.time()

    return np.mean(losses)


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

def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)

def train(args, device, log_dir, rep=None):
    train_dataset = PPIDataset(os.path.join(args.data_dir, 'train'), args.radius, lamb=0.0)
    val_dataset = PPIDataset(os.path.join(args.data_dir, 'val'), args.radius, lamb=0.0)
    test_dataset = PPIDataset(os.path.join(args.data_dir, 'test'), args.radius, lamb=0.0)
    
    train_loader = DataLoader(train_dataset, args.batch_size, num_workers=4)
    val_loader = DataLoader(val_dataset, args.batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, args.batch_size, num_workers=4)
    
    for original, mutated in train_loader:
        dummy_graph = original.clone()
        dummy_graph.x = torch.randn_like(dummy_graph.x)
        dummy_graph.edge_s = torch.randn_like(dummy_graph.edge_s)
        dummy_graph.edge_v = torch.randn_like(dummy_graph.edge_v)
        break

    gnn_model = initialize_model(args.checkpoint, device=device, train=True)
    ff_model = MLPPaired(args.hidden_dim).to(device)
    
    if args.finetune:
        params = [x for x in gnn_model.parameters()] + [x for x in ff_model.parameters()]
    else:
        for param in gnn_model.parameters():
            param.requires_grad = False
        params = ff_model.parameters()

    criterion = nn.BCEWithLogitsLoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    best_val_loss = 999

    for epoch in range(1, args.num_epochs + 1):
        start = time.time()
        train_loss = train_loop(epoch, gnn_model, ff_model, train_loader, criterion, optimizer, device)
        print('validating...')
        val_loss, auroc, auprc, _, _ = test(gnn_model, ff_model, val_loader, criterion, device)
        if val_loss < best_val_loss:
            torch.save({
                'epoch': epoch,
                'gcn_state_dict': gnn_model.state_dict(),
                'ff_state_dict': ff_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
            best_val_loss = val_loss
        elapsed = (time.time() - start)
        print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
        print(f'\tTrain loss {train_loss}, Val loss {val_loss}, Val AUROC {auroc}, Val auprc {auprc}')

    # Evaluate
    train_file = os.path.join(log_dir, f'ppi-rep{rep}.best.train.pt')
    val_file = os.path.join(log_dir, f'ppi-rep{rep}.best.val.pt')
    test_file = os.path.join(log_dir, f'ppi-rep{rep}.best.test.pt')
    cpt = torch.load(os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
    gnn_model.load_state_dict(cpt['gcn_state_dict'])
    ff_model.load_state_dict(cpt['ff_state_dict'])
    _, _, _, y_true_train, y_pred_train = test(gnn_model, ff_model, train_loader, criterion, device)
    torch.save({'targets': y_true_train, 'predictions': y_pred_train}, train_file)
    _, _, _, y_true_val, y_pred_val = test(gnn_model, ff_model, val_loader, criterion, device)
    torch.save({'targets': y_true_val, 'predictions': y_pred_val}, val_file)
    test_loss, auroc, auprc, y_true_test, y_pred_test = test(gnn_model, ff_model, test_loader, criterion, device)
    print(f'\tTest loss {test_loss}, Test AUROC {auroc}, Test auprc {auprc}')
    torch.save({'targets': y_true_test, 'predictions': y_pred_test}, test_file)
    return test_loss, auroc, auprc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--rep', type=int, default=0)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--checkpoint', type=str, default='data/checkpoints/collapse_base.pt')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--radius', type=float, default=10.0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = args.log_dir
        
    print('seed:', args.rep)
    if args.finetune:
        log_dir = os.path.join('logs', f'ppi_{args.checkpoint}')
    else:
        log_dir = os.path.join('logs', f'ppi_fixed_{args.checkpoint}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    np.random.seed(args.rep)
    torch.manual_seed(args.rep)
    train(args, device, log_dir, args.rep)
