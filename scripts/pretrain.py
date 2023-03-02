import torch
import torch.nn.functional as F
import numpy as np
from collapse.byol_pytorch import BYOL
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
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

torch.autograd.set_detect_anomaly(True)

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
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
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
    embeddings_triplets = []
    for i, ((graph_anchor, graph_pos, graph_neg), meta) in enumerate(loader):
        if (graph_anchor == None) or (graph_pos == None) or (graph_neg == None) or (meta == None):
            print(f'validation batch i {i} has None as graph data')
            continue
        else:
            print(f'validation batch i {i} has graph data')
        graph_anchor = graph_anchor.to(device)
        graph_pos = graph_pos.to(device)
        graph_neg = graph_neg.to(device)
        # the numbers in res1, res2 and res3 indicate amino acid residues. There are like 60 residues sampled
        res1, res2, res3 = meta['res_labels']
        cons = meta['conservation'].to(device)
        loss = model(graph_anchor, graph_pos, graph_neg, cons) 
        losses.append(loss.item())
        
        # record std of embeddings (to check for collapsing solution)
        embeddings = model(graph_anchor, graph_pos, graph_neg, return_embedding=True, return_projection=False)
        
        a, b, c = embeddings
        # getting rid of NaNs
        if torch.isnan(a).any() or torch.isnan(b).any() or torch.isnan(c).any():
            print(f"There's NaN in a, b, c embeddings for batch {i}")
            continue
        else:
            print(f"There's no NaN in a, b, c embeddings for batch {i}. a.size {a.size()} -- b.size {b.size()} -- c.size {c.size()}")
            
            
        
        pos_cosine.extend(F.cosine_similarity(a, b).tolist())
        # if graph list has only one graph, the permutation of b values wouldn't work, so we should use c (the negative)
        if b.size(0) > 2:
            neg_cosine.extend(F.cosine_similarity(a, b[torch.randperm(b.size(0))]).tolist())
        elif a.size(0) == c.size(0):
            print(f'graph has only {b.size(0)} positive example, so the negative example will be used with permutation for negative cosine instead of doing permutation of b')
            neg_cosine.extend(F.cosine_similarity(a, c[torch.randperm(c.size(0))]).tolist())
            
        if a.size(0) == c.size(0) and c.size(0) == b.size(0) and (not torch.isnan(c).any()):
            embeddings_triplets.append(torch.cat([embeddings[0], embeddings[1], embeddings[2]]).cpu().detach())
            print(f"using triplicate to find the std in batch {i}")
        else:
            embeddings_triplets.append(torch.cat([embeddings[0], embeddings[1]]).cpu().detach())
            
        embeddings = torch.cat([embeddings[0], embeddings[1]]).cpu().detach()
        
        cls_x.append(embeddings)
        cls_y.append(torch.cat([res1, res2]))
        
   
    cls_x = torch.cat(cls_x)
    cls_y = torch.cat(cls_y)
    emb_norm = cls_x / torch.linalg.norm(cls_x, dim=1).unsqueeze(1)
    
    acc = train_cls(emb_norm, cls_y)
    if len(embeddings_triplets) > 3*i//4:
        embeddings_triplets = torch.cat(embeddings_triplets)
        emb_norm_triplet = embeddings_triplets / torch.linalg.norm(embeddings_triplets, dim=1).unsqueeze(1)
        std = torch.std(emb_norm_triplet, dim=0)
    else: 
        std = torch.std(emb_norm, dim=0)
    
    print(f"The number of dimensions with zero std is {sum(std==0)}")
    
    return np.mean(losses, axis=None), acc, float(torch.mean(std)), np.mean(pos_cosine), np.mean(neg_cosine)

def main():
    
    
    ## print the arguments:
    print("Option '--run_name': ", args.run_name)
    print("Option '--val_dir': ", args.val_dir)
    print("Option '--data_dir': ", args.data_dir)
    print("Option '--checkpoint': ", args.checkpoint)
    print("Option '--lr': ", args.lr)
    print("Option '--env_radius': ", args.env_radius)

    
    """
    LOSSES_FILE_ADDR= '/oak/stanford/groups/rbaltman/alptartici/branch_contrastive/outputContrPretrain/losses.txt'
    
    if os.path.exists(LOSSES_FILE_ADDR):
        os.remove(LOSSES_FILE_ADDR)
    
    RES_SAMPLE_FILE_ADDR = '/oak/stanford/groups/rbaltman/alptartici/branch_contrastive/outputContrPretrain/sampledResids.txt'
    
    if os.path.exists(RES_SAMPLE_FILE_ADDR):
        os.remove(RES_SAMPLE_FILE_ADDR)
        
    PAIR_RESID_FILE_ADDR = '/oak/stanford/groups/rbaltman/alptartici/branch_contrastive/outputContrPretrain/pairResids.txt'
    
    if os.path.exists(PAIR_RESID_FILE_ADDR):
        os.remove(PAIR_RESID_FILE_ADDR)
        
        
    ELEM_FILE_ADDR = '/oak/stanford/groups/rbaltman/alptartici/branch_contrastive/outputContrPretrain/elemContentNew.txt'
    
    if os.path.exists(ELEM_FILE_ADDR):
        os.remove(ELEM_FILE_ADDR)
        
    MSA_FILE_ADDR = '/oak/stanford/groups/rbaltman/alptartici/branch_contrastive/outputContrPretrain/MSAContent.txt'
    if os.path.exists(MSA_FILE_ADDR):
        os.remove(MSA_FILE_ADDR)
        
        
    PATH_FCDD = "/oak/stanford/groups/rbaltman/alptartici/branch_contrastive/outputContrPretrain/failingCDD"
    if os.path.exists(PATH_FCDD):
        os.remove(PATH_FCDD)
        
    FILE_ADDR_ELEM = '/oak/stanford/groups/rbaltman/alptartici/branch_contrastive/outputContrPretrain/batchContent.txt'
    if os.path.exists(FILE_ADDR_ELEM):
        os.remove(FILE_ADDR_ELEM)
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    wandb.init(project="collapse", name=args.run_name, config=vars(args))
    
    #breakpoint()
    #train_dataset = load_dataset(args.data_dir, 'lmdb', transform=CDDTransform(single_chain=True, include_af2=False, env_radius=args.env_radius, num_pairs_sampled=4))
    val_dataset = load_dataset(args.val_dir, 'lmdb', transform=CDDTransform(single_chain=True, include_af2=False, env_radius=args.env_radius, num_pairs_sampled=4, p_hard_negative=1))
    
    dummy_graph = torch.load(os.path.join(os.environ["DATA_DIR"], 'dummy_graph.pt'))
    
    model = BYOL(
        CDDModel(out_dim=args.dim, scatter_mean=True, attn=False, chain_ind=False),
        projection_size=512,
        dummy_graph=dummy_graph,
        hidden_layer=-1,
        use_momentum=(not args.tied_weights)
    ).to(device)
    
    device_ids = [i for i in range(torch.cuda.device_count())]

    # opt = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    opt = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=0.01)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, verbose=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000, eta_min=1e-6)

    if args.checkpoint != "":
        cpt = dict(torch.load(args.checkpoint, map_location=device))
        if not args.tied_weights:
            cpt_model_keys = list(cpt['model_state_dict'].keys())
            for param in cpt_model_keys:
                if 'online_encoder' in param:
                    target_version = 'target_encoder' + param[len('online_encoder'):]
                    if target_version not in cpt_model_keys:
                        cpt['model_state_dict'][target_version] = cpt['model_state_dict'][param]
                
             
        model.load_state_dict(cpt['model_state_dict'])
        # scheduler.load_state_dict(cpt['scheduler_state_dict'])
        opt.load_state_dict(cpt['optimizer_state_dict'])
        args.start_epoch = cpt['epoch']
        
        print('successfully loaded the modified checkpoint that includes target encoder parameters')
        
    if args.parallel:
        # print(f"Using {len(device_ids)} GPUs")
        model = DataParallel(model, device_ids=device_ids)
        #train_loader = DataListLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, collate_fn=NoneCollater(), pin_memory=True, persistent_workers=True)
        val_loader = DataListLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, collate_fn=NoneCollater(), pin_memory=True, persistent_workers=True)
    else:
        #breakpoint()
        #train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=NoneCollater(), pin_memory=True, persistent_workers=True)
        #train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=NoneCollater(), pin_memory=True, persistent_workers=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=NoneCollater(), pin_memory=True, persistent_workers=True)
          
    model.train()
    
    wandb.watch(model)
    MAX_GRAD_NORM = 3
    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    
    """
    epoch = -1
    i = -1
    grad_sum = 0
    grad_max = 0
    grad_min = 0
    GRADSUM_FILE_ADDR = '/oak/stanford/groups/rbaltman/alptartici/branch_contrastive/realPretrainOutput/gradsum.txt'
    file_gradsum = open(GRADSUM_FILE_ADDR, 'a')
    for p in model.parameters():
        if p.requires_grad:
            pgrad = p.grad
            if pgrad is None:
                #pass
                print('at epoch {} and i {}, the param p requires grad but has no grad parameter. \n'.format(epoch, i), file=file_gradsum)
            elif torch.isnan(pgrad).any():
                print('at epoch {} and i {}, the grad for param p has NaN. Pgrad with Nan \n {} PData for this {} \n\n'.format(epoch, i, pgrad, p.data), file=file_gradsum)

            else: 
                #pass
                grad_sum += pgrad.sum()
                grad_max = torch.max(pgrad)
                grad_min = torch.min(pgrad)
            print('at epoch {} and i {}, the grad sum for all parameters is {}, and the max is {} min is {} \n\n'.format(epoch, i, grad_sum, grad_max, grad_min), file=file_gradsum)

    file_gradsum.close()
    """
    best_cos_diff = 0
    best_std = 0
    best_acc = 0
    
    epochNum = float(args.epochs - args.start_epoch)
    
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        # adjust the difficulty of negative examples. make the harder as training progresses
        percent_prog = (epoch - args.start_epoch)/epochNum
        p_hard_neg = np.tanh(4*percent_prog)
        
        train_dataset = load_dataset(args.data_dir, 'lmdb', transform=CDDTransform(single_chain=True, include_af2=False, env_radius=args.env_radius, p_hard_negative=p_hard_neg, num_pairs_sampled=4))
        if args.parallel:
            # print(f"Using {len(device_ids)} GPUs")
            model = DataParallel(model, device_ids=device_ids)
            train_loader = DataListLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, collate_fn=NoneCollater(), pin_memory=True, persistent_workers=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=NoneCollater(), pin_memory=True, persistent_workers=True, drop_last=True)
            

        
        model.train()
        # print(f'EPOCH {epoch+1}:')
        for i, ((graph_anchor, graph_pos, graph_neg), meta) in enumerate(train_loader):
          
            if (graph_anchor == None) or (graph_pos == None) or (graph_neg == None) or (meta == None):
                print(f'training epoch {epoch} batch i {i} has None as graph data')
                continue
            else:
                print(f'training epoch {epoch} batch i {i} has graph data')
            
            """
            print('i =', i)
            if i == 5:
                print('quitting peacefully')
                quit()
            """
            graph_anchor = graph_anchor.to(device)
            graph_pos = graph_pos.to(device)
            graph_neg = graph_neg.to(device)
            cons = meta['conservation'].to(device)
            with torch.cuda.amp.autocast():
                try:
                    loss = model(graph_anchor, graph_pos, graph_neg, loss_weight=cons, return_projection=True)
                except RuntimeError as e:
                    if "CUDA out of memory" not in str(e): 
                        MODPAR_FILE_ADDR = '/oak/stanford/groups/rbaltman/alptartici/branch_contrastive/realPretrainOutput/modelPar.txt'
                        file_modpar = open(MODPAR_FILE_ADDR, 'a')
                        for p in model.parameters():
                            if p.requires_grad:
                                print('p.name {} \np.data \n{}\n\n'.format(p.name, p.data), file=file_modpar)
                        file_modpar.close()  
                        raise(e)
                    torch.cuda.empty_cache()
                    print('Out of Memory error!', flush=True)
                    continue
            # print(loss)
            opt.zero_grad()
            if loss == 0:
                continue
            
            try:
                loss.backward()
            except Exception as e:
                if ('MulBackward0' in str(e)) or ('memory' not in str(e)):
                    MODPAR_FILE_ADDR = '/oak/stanford/groups/rbaltman/alptartici/branch_contrastive/realPretrainOutput/modelPar.txt'
                    file_modpar = open(MODPAR_FILE_ADDR, 'a')
                    for p in model.parameters():
                        if p.requires_grad:
                            pdata = p.data
                            pgrad = p.grad
                            if torch.isnan(pgrad).any():
                                print('gradient of p.name {} at epoch {} index {} contains NaN\n. This is what it looks like {}'.format(p.name, epoch, i, pgrad), file=file_modpar)
                            elif torch.isnan(pdata).any():
                                print('p.name {} at epoch {} index {} has a non-NaN gradient but data contains NaN\n'.format(p.name, epoch, i), file=file_modpar)
                            else:
                                print('p.name {} at epoch {} index {} does not contain NaN'.format(p.name, epoch, i), file=file_modpar)
                            print('p.name {} \np.data \n{}\n\n'.format(p.name, pdata), file=file_modpar)
                    #file_modpar.close()   
                    print(f'at loss backward failure, here is graph_anchor\n {graph_anchor}\n\n and graph_pos\n{graph_pos}\n\n and graph neg\n {graph_neg}\n\nand meta {meta}\n', file=file_modpar)
                    file_modpar.close() 
                    raise Exception('loss backward failed. Look at the modelPar and hVNAN documents to find the source of the error. Full Error message: \n{}'.format(e))
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            opt.step()
            
            grad_sum = 0
            
            GRADSUM_FILE_ADDR = '/oak/stanford/groups/rbaltman/alptartici/branch_contrastive/realPretrainOutput/gradsum.txt'
            file_gradsum = open(GRADSUM_FILE_ADDR, 'a')
            
            for p in model.parameters():
                if p.requires_grad:
                    pgrad = p.grad
                    if pgrad is None:
                        pass
                        #print('at epoch {} and i {}, the param p requires grad but has no grad parameter. \n'.format(epoch, i), file=file_gradsum)
                    elif torch.isnan(pgrad).any():
                        print('at epoch {} and i {}, the grad for param p has NaN. Pgrad with Nan \n {} PData for this {} \n\n'.format(epoch, i, pgrad, p.data), file=file_gradsum)
                        print('graph 1 \n{} \n\n\ngraph 2 \n{}\n\n graph 3 \n{} \n\n\nmeta \n {}\n\n\n\n\n' .format(graph_anchor, graph_pos, graph_neg, meta))
                    else: 
                        pass
                        #grad_sum += pgrad.sum()
                        #grad_max = torch.max(pgrad)
                        #grad_min = torch.min(pgrad)
                    #print('at epoch {} and i {}, the grad sum for all parameters is {}, and the max is {} min is {} \n\n'.format(epoch, i, grad_sum, grad_max, grad_min), file=file_gradsum)
            
            file_gradsum.close()
                    
 
            if not args.tied_weights:
                model.update_moving_average(epoch) # update moving average of target encoder
            wandb.log({'loss': loss.item()})
            print(f'Epoch {epoch}: Iteration {i}: Loss: {loss.item()}')
        
        val_loss, acc, std, pos_cosine, neg_cosine = evaluate(val_loader, model, device)
        wandb.log({'epoch': epoch, 'val_loss': val_loss, 'aa_knn_acc': acc, 'std': std, 'pos_cosine': pos_cosine, 'neg_cosine': neg_cosine})
        
        # save your improved network only if it's an improvement
        if ((pos_cosine - neg_cosine) > best_cos_diff) and (std > best_std) and (acc > best_acc):
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
            
            best_cos_diff = pos_cosine - neg_cosine
            best_std = std
            best_acc = acc
            
        # scheduler.step(val_loss)
        

def train_cls(x, y):
    mod = LogisticRegression(max_iter=100, solver="liblinear")
    scores = cross_val_score(mod, x, y, cv=4)
    return np.mean(scores)

if __name__ == "__main__":
    main()
