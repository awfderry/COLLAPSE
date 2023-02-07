import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_geometric.utils import softmax
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from gvp import GVP, GVPConvLayer, LayerNorm
import inspect
#import matplotlib.pyplot as plt


# _NUM_ATOM_TYPES = 9
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
    def __init__(self, num_rbf=16, out_dim=512, scatter_mean=True, attn=False, distance_based=False, chain_ind=False):
        
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
        
        
        #print('batch.dists\n', batch.dists, '\n\n\n')
        
        # print('vars(batch[0])', vars(batch[0]))
        #print('batch[0].dists', batch[0].dists)
        #print('batch[0].distnodes', batch[0].distnodes)
        
        #print('getting the members of the batch object\n')
        dictOfAttr = vars(batch)
        torch.set_printoptions(threshold=999)
        
        """
        for key in dictOfAttr:
            if torch.is_tensor(dictOfAttr[key]):
                print('attr ', key, 'with dimensions ', dictOfAttr[key].shape, '\n')
            else:
                print('attr ', key)
            print(dictOfAttr[key], '\n\n')
        print('\nthese are all the members of batch')
        
        """
        
        if '_store' in dictOfAttr:
            if 'dist_to_ctr' in dictOfAttr['_store']:
                #print('finally has the desired distance attributes')
                distances = batch._store['dist_to_ctr']
                #print("batch._store['dist_to_ctr']", distances)
                #print('shape of dist_to_ctr ', distances.shape)
                node_index = batch._store['node_index']
                #print("batch._store['node_index']", node_index)
                #print('shape of node_index ', node_index.shape)
                #print("mean distance ", torch.mean(distances))
                #print("min distance ", torch.min(distances))
                #print("max distance ", torch.max(distances))
                #print("median distance ", torch.median(distances))
                if self.scatter_mean or self.attn:
                    self.scatter_mean = False
                    self.attn = False
                    self.distance_based = True
                #print('SUCCESS')
                
                
                """
                fig1 = plt.figure()
                plt.bar(distances.flatten())
                plt.xlabel('distances (angstrom)')
                plt.ylabel('atom count')
                plt.savefig('../outputPretrain/distance-distribution-bar')
                
                fig2 = plt.figure()
                plt.boxplot(distances.flatten())
                plt.xlabel("distances (angstrom)")
                #plt.ylabel("Frequency")
                plt.savefig('../outputPretrain/distance-distribution-box')
        
                """
      
                """
                DISTS_FILE_ADDR = '/oak/stanford/groups/rbaltman/alptartici/COLLAPSE/outputPretrain/dists.txt'
                file_dists = open(DISTS_FILE_ADDR, 'w')
                print(distances, file=file_dists)
                file_dists.close()
                
                
                DIMENSIONS_FILE_ADDR = '/oak/stanford/groups/rbaltman/alptartici/COLLAPSE/outputPretrain/dimensions.txt'
                file_dimensions = open(DIMENSIONS_FILE_ADDR, 'a')
                print('distances dimensions\n', file=file_dimensions)
                print(distances.shape, file=file_dimensions)
                print('\n\n\n', file=file_dimensions)
                file_dimensions.close()
                """
        
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
            for subtensor in h_V:
                if torch.isnan(subtensor).any():
                    print('h_V tensor w NaN inside CDDModel forward() \n{}'.format(h_V))
                    raise Exception('There is NaN in h_V inside CDDModel forward before layer \n{}\n'.format(layer))
            h_V = layer(h_V, batch.edge_index, h_E)
            #breakpoint()

        
        out = self.W_out(h_V)
        
        if torch.isnan(out.any()):
            print('out tensor inside CDDModel forward() {}\n'.format(out))
            raise Exception('NaN as the output of GNN layers')
        
        ## make sure out does not explode
        out = torch.clamp(out, min=-10, max=10)
        
        """
        print('batch_id\n', batch_id, '\n\n\n')
        print('out shape ', out.shape, '\n\n')
        print('out content\n', out, '\n\n')
        """
        
        if no_pool:
            return out

        elif self.scatter_mean: 
            #print('self.scatter_mean is executed inside CDDModel forward')
            out = torch_scatter.scatter_mean(out, batch_id, dim=0)
            """
            DIMENSIONS_FILE_ADDR = '/oak/stanford/groups/rbaltman/alptartici/COLLAPSE/outputPretrain/dimensions.txt'
            file_dimensions = open(DIMENSIONS_FILE_ADDR, 'a')
            print('out (the correct one done for dummy_graph) dimensions\n',file = file_dimensions)
            print(out.shape, file=file_dimensions)
            print('\n\n\n', file=file_dimensions)
            file_dimensions.close()
            """
        elif self.attn: 
            attn = self.attn_nn(out).view(-1, 1)
            attn = softmax(attn, batch_id)
            out = torch_scatter.scatter_mean(attn * out, batch_id, dim=0)
        elif self.distance_based:
            #print('out.shape ', out.shape, '\n')
            # the batch contains the graph
            # get the graph dists sum
            param_a = 2
            #param_a.requires_grad = False
            param_b = 0.05
            #param_b.requires_grad = False
            distances = torch.clamp(distances, min=-0.001, max=20)
            distances.requires_grad = False
            
            dist_weight = torch.clamp(param_a - torch.exp(param_b * distances),
                                      min = 0,
                                      max = 1)
            dist_weight.requires_grad = False

            # get the sum of transformed distances weights per batch
            
            batch_dist_weight_sums = torch_scatter.scatter_add(dist_weight, batch_id, dim=0)
           
            # normalize the transformed distance weights so they add up to 1
            # this is necessar
            normalized_distance_weights = dist_weight.clone().detach().requires_grad_(False)
            for i in range(dist_weight.shape[0]):
                # get what batch the distance corresponds to
                corresponding_batch = batch_id[i]
                # get the sum of distances for that batch
                sud = batch_dist_weight_sums[corresponding_batch]
                # divide by the some of distances
                normalized_distance_weights[i] = dist_weight[i]/sud
            
            
            normalized_distance_weights = torch.clamp(normalized_distance_weights, min=0, max=1)
            
            
            # convert it from 1D to 2D to make dimensions compatible
            normalized_distance_weights = normalized_distance_weights.view(-1,1)
            normalized_distance_weights.requires_grad = False
        
            """
            DIMENSIONS_FILE_ADDR = '/oak/stanford/groups/rbaltman/alptartici/COLLAPSE/outputPretrain/dimensions.txt'
            file_dimensions = open(DIMENSIONS_FILE_ADDR, 'a')
            print('normalized_distance_weights dimensions\n', file=file_dimensions)
            print(normalized_distance_weights.shape, file=file_dimensions)
            print('\n\n\n', file=file_dimensions)
            print('out (after layers, before scatter mean) dimensions\n', file=file_dimensions)
            print(out.shape, file=file_dimensions)
            print('\n\n\n', file=file_dimensions)
            print('batch_id dimensions\n', file=file_dimensions)
            print(batch_id.shape, file=file_dimensions)
            print('\n\n\n', file=file_dimensions)
            print('normalized_distance_weights.T dimensions\n', file=file_dimensions)
            print(normalized_distance_weights.T.shape, file=file_dimensions)
            print('\n\n\n', file=file_dimensions)
            file_dimensions.close()
        
        
            # sanity check for the new weights
            weights_FILE_ADDR = '/oak/stanford/groups/rbaltman/alptartici/COLLAPSE/outputPretrain/weights.txt'
            file_weights = open(weights_FILE_ADDR, 'w')
            print(normalized_distance_weights, file=file_weights)
            print(torch.sum(normalized_distance_weights), file=file_weights)
            file_weights.close()
            """
            
            out_before_scatter = normalized_distance_weights * out
            out = torch_scatter.scatter_mean(out_before_scatter, batch_id, dim=0)
            
          
            
            #print('SUCCESS, THIS CODE RAN SO YOU CAN IMPLEMENT DISTANCE')
            
            """
            DIMENSIONS_FILE_ADDR = '/oak/stanford/groups/rbaltman/alptartici/COLLAPSE/outputPretrain/dimensions.txt'
            file_dimensions = open(DIMENSIONS_FILE_ADDR, 'a')
            print('out (output of distance based weighting) dimensions\n', file=file_dimensions)
            print(out.shape, file=file_dimensions)
            print('\n\n\n', file=file_dimensions)
            file_dimensions.close()
            """
            
        
        #print('finalized the out inside CDDModel forward')
        
        

        return out
    
    
    def forward_dummy(self, batch, no_pool=False):
        '''
        Forward pass which can be adjusted based on task formulation.
        
        :param batch: `torch_geometric.data.Batch` with data attributes
                      as returned from a BaseTransform
        '''
        
        # running for the dummy graph
        
        
        
        
        #print('testing the additional features')
        #print("batch._store['dist_to_ctr']", batch._store['dist_to_ctr'])
        #print("batch._store['node_index']", batch._store['node_index'])
        
   
      
        
        if self.scatter_mean and self.attn:
            raise Exception('only one of scatter_mean and attn can be used at once')
        
        h_V = self.embed(batch.atoms)
        if self.chain_ind:
            h_V = torch.cat((h_V, batch.chain_ind.unsqueeze(1)), dim=-1)
        h_E = (batch.edge_s, batch.edge_v)
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        
        batch_id = batch.batch
        print('batch_id', batch_id, '\n\n\n')
        
        for layer in self.layers:
            h_V = layer(h_V, batch.edge_index, h_E)

        out = self.W_out(h_V)
        if no_pool:
            return out

        elif self.scatter_mean: 
            print('self.scatter_mean is executed inside CDDModel forward')
            out = torch_scatter.scatter_mean(out, batch_id, dim=0)
        elif self.attn: 
            attn = self.attn_nn(out).view(-1, 1)
            attn = softmax(attn, batch_id)
            out = torch_scatter.scatter_mean(attn * out, batch_id, dim=0)
        elif self.distance_based:
            # the batch contains the graph
            # get the graph dists
            # convert dists to weights
            # do weighted averages
            pass
        
        print('finalized the out inside CDDModel forward')

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
    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.5):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dims[0]))
        if len(hidden_dims) > 1:
            for i, h in enumerate(hidden_dims[1:]):
                self.layers.append(nn.Linear(hidden_dims[i], h))
        self.layers.append(nn.Linear(hidden_dims[-1], out_dim))
        self.dropout = dropout

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)
        return x

class LSTM(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional=True, dropout=0.5):
        super().__init__()
        self.lstm = nn.GRU(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            dropout=dropout,
                            bidirectional=bidirectional,
                            batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def attention(self, output, hidden, mask):	
        attn_weights = torch.bmm(output, hidden.unsqueeze(2)).squeeze(2)
        attn_weights[~mask] = float('-inf')
        attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(output.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, embeddings, lengths):
        packed_embedded = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False) 
        
        packed_output, (hidden) = self.lstm(packed_embedded)
        # output, lens = pad_packed_sequence(packed_output, batch_first=True)
        # mask = torch.arange(output.size(1))[None, :] < lens[:, None]
        hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # hidden_attn = self.attention(output, hidden_cat, mask)
        dense1 = self.fc1(hidden_cat)
        drop = self.dropout(dense1)
        rel = self.relu(drop)
        preds = self.fc2(rel)
        return preds

class AttentionAggregator(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.attn_nn = nn.Linear(embedding_dim, 1)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings, lengths):
        # embeddings = b x L x 512
        lengths = torch.tensor(lengths)
        mask = torch.arange(embeddings.size(1))[None, :] < lengths[:, None]
        attn = self.attn_nn(embeddings).squeeze() # b x L
        attn[~mask] = float('-inf')
        attn = F.softmax(attn, 1)
        attn_out = torch.bmm(embeddings.transpose(1,2), attn.unsqueeze(2)).squeeze(2) # b x 512
        x = self.fc1(attn_out)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        preds = self.fc3(x)
        return preds
    

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
