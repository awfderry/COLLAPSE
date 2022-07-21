import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_geometric.utils import softmax

from gvp import GVP, GVPConvLayer, LayerNorm


_NUM_ATOM_TYPES = 13
_DEFAULT_V_DIM = (100, 16)
_DEFAULT_E_DIM = (32, 1)


class CDDModel(nn.Module):
    '''
    Adapted from https://github.com/drorlab/gvp-pytorch
    
    
    A base 5-layer GVP-GNN for all ATOM3D tasks, using GVPs with 
    vector gating as described in the manuscript. Takes in atomic-level
    structure graphs of type `torch_geometric.data.Batch`
    and returns a single scalar.
    
    This class should not be used directly. Instead, please use the
    task-specific models which extend BaseModel. (Some of these classes
    may be aliases of BaseModel.)
    
    :param num_rbf: number of radial bases to use in the edge embedding
    '''
    def __init__(self, num_rbf=16, out_dim=512, scatter_mean=True, attn=False, chain_ind=False):
        
        super().__init__()
        self.scatter_mean = scatter_mean
        self.attn = attn
        self.chain_ind = chain_ind
        
        in_dim = _NUM_ATOM_TYPES
        if self.chain_ind:
            in_dim += 1
        
        activations = (F.relu, None)
        
        self.embed = nn.Embedding(in_dim, in_dim)
        
        self.W_e = nn.Sequential(
            LayerNorm((num_rbf, 1)),
            GVP((num_rbf, 1), _DEFAULT_E_DIM, 
                activations=(None, None), vector_gate=True)
        )
        
        self.W_v = nn.Sequential(
            LayerNorm((in_dim, 0)),
            GVP((in_dim, 0), _DEFAULT_V_DIM,
                activations=(None, None), vector_gate=True)
        )
        
        self.layers = nn.ModuleList(
            GVPConvLayer(_DEFAULT_V_DIM, _DEFAULT_E_DIM, 
                         activations=activations, vector_gate=True) 
            for _ in range(5)
        )
        
        ns, _ = _DEFAULT_V_DIM
        self.W_out = nn.Sequential(
            LayerNorm(_DEFAULT_V_DIM),
            GVP(_DEFAULT_V_DIM, (out_dim, 0), 
                activations=activations, vector_gate=True)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(ns, 2 * ns), nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(2 * ns, out_dim)
        )
        
        if self.attn:
            self.attn_nn = nn.Linear(out_dim, 1)
            
    
    def forward(self, batch, no_pool=False):
        '''
        Forward pass which can be adjusted based on task formulation.
        
        :param batch: `torch_geometric.data.Batch` with data attributes
                      as returned from a BaseTransform
        '''
        if self.scatter_mean and self.attn:
            raise Exception('only one of scatter_mean and attn can be used at once')
        
        h_V = self.embed(batch.atoms)
        if self.chain_ind:
            h_V = torch.cat((h_V, batch.chain_ind.unsqueeze(1)), dim=-1)
        h_E = (batch.edge_s, batch.edge_v)
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        
        batch_id = batch.batch
        
        for layer in self.layers:
            h_V = layer(h_V, batch.edge_index, h_E)

        out = self.W_out(h_V)
        if no_pool:
            return out

        elif self.scatter_mean: 
            out = torch_scatter.scatter_mean(out, batch_id, dim=0)
        elif self.attn: 
            attn = self.attn_nn(out).view(-1, 1)
            attn = softmax(attn, batch_id)
            out = torch_scatter.scatter_mean(attn * out, batch_id, dim=0)

        return out
   
    
class MSPModel(CDDModel):
    '''
    from https://github.com/drorlab/gvp-pytorch/blob/main/gvp/atom3d.py
    '''
    def __init__(self, ns=512, **kwargs):
        super().__init__(**kwargs)
        self.dense = nn.Sequential(
            nn.Linear(2 * ns, 4 * ns), nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(4 * ns, 1)
        )
        
    def forward(self, orig, mut):        
        out1, out2 = map(self._gnn_forward, [orig, mut])
        out = torch.cat([out1, out2], dim=-1)
        out = self.dense(out)
        return torch.sigmoid(out).squeeze(-1)
    
    def _gnn_forward(self, graph):
        out = super().forward(graph, no_pool=True)
        out = out * graph.node_mask.unsqueeze(-1)
        out = torch_scatter.scatter_add(out, graph.batch, dim=0)
        count = torch_scatter.scatter_add(graph.node_mask, graph.batch)
        return out / count.unsqueeze(-1)
    

class MLPPaired(torch.nn.Module):
    def __init__(self, in_dim=512, hidden_dim=512 * 4):
        super(MLPPaired, self).__init__()
        self.fc1 = nn.Linear(in_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, input1, input2):
        x = torch.cat((input1, input2), dim=1)
        x = self.fc1(x)
        # x = self.bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.50, training=self.training)
        x = self.fc2(x)
        return x.view(-1)


class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.50, training=self.training)
        x = self.fc2(x)
        return x
    

if __name__ == "__main__":
    from data import CDDTransform
    from atom3d.datasets import load_dataset
    from torch_geometric.data import DataLoader
    dataset = load_dataset('lmdb/cdd_pdb_dataset', 'lmdb', transform=CDDTransform())
    loader = DataLoader(dataset, batch_size=2)
    model = CDDModel()
    print(model)
    for graph in loader:
        output = model(graph)
        break
