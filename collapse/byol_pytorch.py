"""
The functions and classes in this file are modified from https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py
"""

import copy
from functools import wraps
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import SoftMarginLoss

# helper functions

def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss fn

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    
    #diff = x - y
    #return torch.clamp(torch.abs(diff), min=0, max=2)
    #return 2 - 2 * (x * y).sum(dim=-1)
    cosDist = 1 - (x * y).sum(dim=-1)
    return torch.clamp(cosDist, min=0, max=2)
        

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new, current_epoch):
        if old is None:
            return new
        self.beta = 1 - (1 - self.beta) * (math.cos(math.pi * current_epoch / 5000) + 1) / 2
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model, current_epoch):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight, current_epoch)

# MLP class for projector and predictor

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer=-1, extract_center=True, dense=False):
        super().__init__()
        self.net = net
        self.layer = layer
        self.dense = dense
        self.extract_center = extract_center

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection = True):
        representation = self.get_representation(x)

        if not return_projection:
            return representation, None

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation

# main class

class BYOL(nn.Module):
    def __init__(
        self,
        net,
        dummy_graph,
        hidden_layer = -1,
        projection_size = 512,
        projection_hidden_size = 4096,
        augment_fn = None,
        augment_fn2 = None,
        moving_average_decay = 0.99,
        use_momentum = True,
        extract_center=True, 
        dense=False
    ):
        super().__init__()
        self.net = net

        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer, dense=dense)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)
        self.loss = SoftMarginLoss()

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(dummy_graph, dummy_graph, dummy_graph, 1)

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self, current_epoch):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder, current_epoch)

    def forward(
        self,
        graph_anchor, graph_pos, graph_neg, loss_weight=1.0,
        return_embedding=False,
        return_projection=True
    ):
        # assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            anc_emb = self.online_encoder(graph_anchor, return_projection=return_projection)[0]
            pos_emb = self.online_encoder(graph_pos, return_projection=return_projection)[0]
            neg_emb = self.online_encoder(graph_neg, return_projection=return_projection)[0]
            return anc_emb, pos_emb, neg_emb

        online_proj_anchor, _ = self.online_encoder(graph_anchor, return_projection=return_projection)
        online_proj_pos, _ = self.online_encoder(graph_pos, return_projection=return_projection)
        online_proj_neg, _ = self.online_encoder(graph_neg, return_projection=return_projection)

        online_pred_anchor = self.online_predictor(online_proj_anchor)
        online_pred_pos = self.online_predictor(online_proj_pos)
        online_pred_neg = self.online_predictor(online_proj_neg)
        
        if torch.isnan(online_proj_anchor).any() or torch.isnan(online_pred_pos).any() or torch.isnan(online_pred_neg).any() or (online_pred_anchor.nelement() == 0) or (online_pred_pos.nelement() == 0) or (online_pred_neg.nelement() == 0):
            return 0

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_anchor, _ = target_encoder(graph_anchor, return_projection=return_projection)
            target_proj_pos, _ = target_encoder(graph_pos, return_projection=return_projection)
            target_proj_neg, _ = target_encoder(graph_neg, return_projection=return_projection)
            target_proj_anchor.detach_()
            target_proj_pos.detach_()
            target_proj_neg.detach_()
        
        if torch.isnan(target_proj_pos).any() or torch.isnan(target_proj_anchor).any() or torch.isnan(target_proj_neg).any() or (target_proj_anchor.nelement() == 0) or (target_proj_pos.nelement() == 0) or (target_proj_neg.nelement() == 0):
            return 0
        
        # if the embeddings are very different, dist value is high
        dist_pos_1 = loss_fn(online_pred_anchor, target_proj_pos.detach())
        dist_pos_2 = loss_fn(online_pred_pos, target_proj_anchor.detach())
        dist_pos_combined = dist_pos_1 + dist_pos_2
        
        dist_neg_1 = loss_fn(online_pred_anchor, target_proj_neg.detach())
        dist_neg_2 = loss_fn(online_pred_neg, target_proj_anchor.detach())
        dist_neg_combined = dist_neg_1 + dist_neg_2
        
        # this is an arbitrary parameter to crank up the loss
        """
        LOSS_WEIGHT_MULTIPLIER_HPARAM = 3 
        exp_param = LOSS_WEIGHT_MULTIPLIER_HPARAM * loss_weight
        loss = torch.log(1 + torch.exp(exp_param * (dist_pos_combined - dist_neg_combined)))
        """
        
        len_dist_pos = dist_pos_combined.nelement()
        len_dist_neg = dist_neg_combined.nelement()
        
        if len_dist_pos == 0 or len_dist_neg == 0 or len_dist_pos != len_dist_neg:
            return 0
        
        
        """
        embeddings_pos_neg = torch.cat([online_pred_pos, online_pred_neg])
        std_pos_neg = torch.std(embeddings_pos_neg, dim=0)
        
        # we want the mean of the std to be high. So we want to minimize the negative of the mean
        mean_std = torch.mean(std_pos_neg)
        # we want the std of std to be low. So we want to minimize positive std_std.
        std_std = torch.std(std_pos_neg)
        """
        l1_dist_pos_neg = torch.clamp(torch.abs(online_pred_pos - online_pred_neg), min=0, max=2)
        mean_l1 = torch.mean(l1_dist_pos_neg)
        std_l1 = torch.std(l1_dist_pos_neg)
        
        
        # MARGIN DIST OF 0.5
        MARGIN = 6*torch.ones_like(dist_pos_combined, requires_grad=False)
        yLabel = -1*torch.ones_like(dist_pos_combined, requires_grad=False)
        # what we want to minimize: dist pos (0-2), std_l1 (0-2)
        # what we want to maximize: dist_neg (0-2), mean_l1 (0-2)
        loss_to_minimize = dist_pos_combined - dist_neg_combined + MARGIN + 2*std_l1 - 2*mean_l1
        loss = self.loss((loss_to_minimize), yLabel)
        #loss = torch.log(1 + torch.exp(dist_pos_combined - dist_neg_combined))
        #loss = torch.clamp(loss, min=-0.5, max=10)
        
        
        """
        LOSSES_FILE_ADDR= '/oak/stanford/groups/rbaltman/alptartici/branch_contrastive/outputContrPretrain/losses.txt'
        file_losses = open(LOSSES_FILE_ADDR, 'a')
        print('losses and distances\n', file=file_losses)
        print('dist_pos_combined {}'.format(dist_pos_combined), file=file_losses)
        print('dist_neg_combined {}'.format(dist_neg_combined), file=file_losses)
        print('loss {}'.format(loss), file=file_losses)
        print('\n\n\n', file=file_losses)
        file_losses.close()
        """
        
        return torch.clamp(loss.nanmean(), min=-1, max=8)
