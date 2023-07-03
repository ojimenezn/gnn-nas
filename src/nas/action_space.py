from torch.nn import ModuleList
import numpy as np
from typing import Tuple, Union

import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
from torch.nn import GRU, Linear, ReLU, Tanh, Sigmoid, LeakyReLU, Sequential, BatchNorm1d, Linear
from torch_geometric.loader import DataLoader
from torch_geometric.nn import  global_mean_pool, global_max_pool, global_add_pool, CGConv, GraphConv, GCNConv, GATConv, GPSConv, GINEConv, BatchNorm, LayerNorm
from torch_geometric.data import Dataset, Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor

from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from IPython.display import Image
import imageio
from google.colab import drive
drive.mount('/drive')

class NasGNN(nn.Module):
    def __init__(self, blocks, aggrs):

        """
        Constructor for a NAS-generated GNN
        """
        super(NasGNN, self).__init__()


        self.dim = dim
        self.lin0 = torch.nn.Linear(1, dim)

        self.convs = ModuleList()
        self.acts = ModuleList()
        self.norms = ModuleList()

        max_blks = 3
        for i in range(max_blks):
          conv = blocks[i]
          conv.aggr = aggrs[i]
          norm = blocks[3]
          act = blocks[i+4]
          self.convs.append(conv)
          self.norms.append(norm)
          self.acts.append(act)
        
        self.pooling = blocks[-1]
        #self.skip = skip
        self.lin1 = blocks[-2]

        self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, data):

        """
        Skeleton for forward pass. 
        """

        x = data.x
        out = F.relu(self.lin0(data.x))

        for i in range(len(self.convs)):
          act = self.acts[i]
          conv = self.convs[i]
          norm = self.norms[i]
          out = act(conv(out, data.edge_index, data.edge_attr))
          out = norm(out)

        if self.pooling == str(list(params.keys()).index('global_add') - 1): out = global_add_pool(out, data.batch)
        elif self.pooling == str(list(params.keys()).index('global_mean') - 1): out = global_mean_pool(out, data.batch)
        else: out = global_max_pool(out, data.batch)

        out = F.relu(self.lin1(out))
        out = self.lin2(out)

        return out.view(-1)

class NasGNN(nn.Module):
    def __init__(self, blocks, blocks2, aggrs, aggrs2, aggr_type):

        """
        Note: NOT the final constructor. This is more of a manual 
        sanity check for 2 block architectures to debug invalid configurations
        """
        super(NasGNN, self).__init__()

        # Constructor for only 2 block architectures
        self.dim = dim
        self.lin0 = torch.nn.Linear(1, dim)
        self.conv1 = blocks[0] 
        self.aggr1 = aggrs[0]
        self.conv1.aggr = aggrs[0]
        self.conv1_act = blocks[4] 
        self.conv2 = blocks[1] 
        self.aggr2 = aggrs[1]
        self.conv2.aggr = aggrs[1]
        self.conv2_act = blocks[5] 
        self.conv3 = blocks[2] 
        self.aggr3 = aggrs[2]
        self.conv3.aggr = aggrs[2]
        self.conv3_act = blocks[6]
        self.norm = blocks[3]
        self.pooling = blocks[-1]
        self.lin1 = blocks[-2]

        self.aggr_type = aggr_type

        ##### Block 2
        self.conv4 = blocks2[0] 
        self.aggr4 = aggrs2[0]
        self.conv4.aggr = aggrs2[0]
        self.conv4_act = blocks2[4] 
        self.conv5 = blocks2[1] 
        self.aggr5 = aggrs2[1]
        self.conv5.aggr = aggrs2[1]
        self.conv5_act = blocks2[5] 
        self.conv6 = blocks2[2] 
        self.aggr6 = aggrs2[2]
        self.conv6.aggr = aggrs2[2]
        self.conv6_act = blocks2[6]
        self.norm2 = blocks2[3]
        self.pooling2 = blocks2[-1]
        self.lin12 = blocks2[-2]

        self.lin2 = torch.nn.Linear(dim, 1) # if sum blocks
        self.lin3 = torch.nn.Linear(dim*2, 1) # if concat 2 blocks


    def forward(self, data):

        """
        Note: Manual too. Not generic for smaller architectures
        """

        #### Block 1
        out1 = F.relu(self.lin0(data.x))

        out1 = self.conv1_act(self.conv1(out1, data.edge_index, data.edge_attr))
        out1 = self.norm(out1)
        out1 = self.conv2_act(self.conv2(out1, data.edge_index, data.edge_attr))
        out1 = self.norm(out1)
        out1 = self.conv3_act(self.conv3(out1, data.edge_index, data.edge_attr))
        out1 = self.norm(out1)

        if self.pooling == str(list(params.keys()).index('global_add') - 1): out1 = global_add_pool(out1, data.batch)
        elif self.pooling == str(list(params.keys()).index('global_mean') - 1): out1 = global_mean_pool(out1, data.batch)
        else: out1 = global_max_pool(out1, data.batch)

        out1 = F.relu(self.lin1(out1))

        #### Block 2
        out2 = F.relu(self.lin0(data.x))

        out2 = self.conv4_act(self.conv4(out2, data.edge_index, data.edge_attr))
        out2 = self.norm2(out2)
        out2 = self.conv5_act(self.conv5(out2, data.edge_index, data.edge_attr))
        out2 = self.norm2(out2)
        out2 = self.conv6_act(self.conv6(out2, data.edge_index, data.edge_attr))
        out2 = self.norm2(out2)

        if self.pooling2 == str(list(params.keys()).index('global_add') - 1): out2 = global_add_pool(out2, data.batch)
        elif self.pooling2 == str(list(params.keys()).index('global_mean') - 1): out2 = global_mean_pool(out2, data.batch)
        else: out2 = global_max_pool(out2, data.batch)

        out2 = F.relu(self.lin12(out2))

        if self.aggr_type == 'sum': 
          out = out1 + out2
          out = self.lin2(out) # add
        else: 
          out = torch.cat((out1, out2), dim=1)
          out = self.lin3(out) # concatenation

        return out.view(-1)


def get_action_space(cur_len, scores):

    """
    Returns valid actions in the action space given current sequence length 
    by masking invalid actions (https://arxiv.org/pdf/2006.14171.pdf).
    In this version of the code I was doing some further tests, but for the 
    actual experiments we only masked the hyperparameters since those are necessary
    to be generated always during the first 3 sequence values. The actions in the 
    actual layer logic were not masked to encourage exploration (which is why a lot 
    of the architectures failed). In this setup we have more control and generates 
    more valid architectures.
    """

    mask = -1e8

    # First actions correspond to hyperparameters
    if cur_len == 0:
      for non_lr_idx in non_lr_idxs:
        scores[:,non_lr_idx] = mask 

    if cur_len == 1:
      for non_batch_size_idx in non_batch_size_idxs:
        scores[:,non_batch_size_idx] = mask

    if cur_len == 2:
      for non_dim_idx in non_dims_idxs:
        scores[:,non_dim_idx] = mask 

    # Intermediate actions correspond to actual architecture 
    # We limit up to a max of 3 GNN layers for computational  
    # cost but certainly this is not a restriction 
    if cur_len >= 3 and cur_len <= 5:
      for non_gnn_layer_idx in non_gnn_layers_idxs:
        scores[:,non_gnn_layer_idx] = mask

    if cur_len >= 6 and cur_len <= 8:
      for non_aggr_idx in non_aggr_idxs:
        scores[:,non_aggr_idx] = mask
    
    if cur_len == 9:
      for non_normalization_idx in non_normalization_idxs:
        scores[:,non_normalization_idx] = mask

    # Activation functions for the up to 3 GNN layers
    if cur_len >= 10 and cur_len <= 12:
      for non_activation_idx in non_activation_idxs:
        scores[:,non_activation_idx] = mask

    # Linear layer at end of architecture since we want energy
    if cur_len == 13:
      for non_linear_layers_idx in non_linear_layers_idxs:
        scores[:,non_linear_layers_idx] = mask
    
    # Pooling
    if cur_len == 14:
      for non_pooling_idx in non_pooling_idxs:
        scores[:,non_pooling_idx] = mask
    
    if cur_len == 15:
      for non_block_idx in non_block_idxs:
        scores[:,non_block_idx] = mask

    ### Block 2
    if cur_len == 16:
      for non_lr_idx in non_lr_idxs:
        scores[:,non_lr_idx] = mask 

    if cur_len == 17:
      for non_batch_size_idx in non_batch_size_idxs:
        scores[:,non_batch_size_idx] = mask

    if cur_len == 18:
      for non_dim_idx in non_dims_idxs:
        scores[:,non_dim_idx] = mask 

    if cur_len >= 19 and cur_len <= 21:
      for non_gnn_layer_idx in non_gnn_layers_idxs:
        scores[:,non_gnn_layer_idx] = mask

    if cur_len >= 22 and cur_len <= 23:
      for non_aggr_idx in non_aggr_idxs:
        scores[:,non_aggr_idx] = mask
    
    if cur_len == 24:
      for non_normalization_idx in non_normalization_idxs:
        scores[:,non_normalization_idx] = mask

    if cur_len >= 25 and cur_len <= 27:
      for non_activation_idx in non_activation_idxs:
        scores[:,non_activation_idx] = mask

    if cur_len == 28:
      for non_linear_layers_idx in non_linear_layers_idxs:
        scores[:,non_linear_layers_idx] = mask
    
    if cur_len == 29:
      for non_pooling_idx in non_pooling_idxs:
        scores[:,non_pooling_idx] = mask

    return scores

def get_model(generated, device):

    """
    Get torch.nn.Module object from generated sequence.
    Note to self: this is ugly ugly code. Make more elegant eventually.
    """

    original_generated = generated.copy()
    generated = generated[:15]

    d = 1
    dim = int(list(params.keys())[list(params.values()).index(int(generated[2]))])
    lyrs = generated[3:6]
    aggrs = generated[6:9]
    aggrs = [next(key for key, value in params.items() if str(value) == item) if item.isdigit() and int(item) in params.values() else item for item in aggrs]
    norm = generated[9:10]
    acts = generated[10:13]
    pooling = generated[-1]
    #skip = generated[-1]
    generated_list = lyrs + norm + acts

    blocks1 = []
    for component in generated_list:
        if component in architecture_dict:
            block_type = architecture_dict[component]
            if block_type == Linear:
                block = block_type(in_features=dim, out_features=dim)  
            elif block_type == CGConv:
                block = block_type(dim, d) # aggr later inside init       
            elif block_type == GraphConv or block_type == GATConv or block_type == GCNConv:
                block = block_type(dim, dim) 
            elif block_type == BatchNorm or block_type == LayerNorm:
                block = block_type(dim) 
            elif block_type == GPSConv:
                nn = Sequential(
                    Linear(dim, dim),
                    ReLU(),
                    Linear(dim, dim),
                )
                block = GPSConv(dim, GINEConv(nn), heads=4, attn_dropout=0.5)
            else:
                block = block_type()
            blocks1.append(block)
        else:
            raise ValueError(f"Component '{component}' not found in architecture dictionary.")

    aggr_type = original_generated[15]
    if aggr_type == '29': aggr_type = 'concat'
    else: aggr_type = 'sum'
    generated2 = original_generated[16:]

    d = 1
    dim2 = int(list(params.keys())[list(params.values()).index(int(generated2[2]))])
    lyrs2 = generated2[3:6]
    aggrs2 = generated2[6:9]
    aggrs2 = [next(key for key, value in params.items() if str(value) == item) if item.isdigit() and int(item) in params.values() else item for item in aggrs2]
    norm2 = generated2[9:10]
    acts2 = generated2[10:13]
    pooling2 = generated2[-1]
    generated_list_2 = lyrs2 + norm2 + acts2

    blocks2 = []
    for component in generated_list_2:
        if component in architecture_dict:
            block_type = architecture_dict[component]
            if block_type == Linear:
                block = block_type(in_features=dim, out_features=dim)  
            elif block_type == CGConv:
                block = block_type(dim, d) # aggr later inside init       
            elif block_type == GraphConv or block_type == GATConv or block_type == GCNConv:
                block = block_type(dim, dim) 
            elif block_type == BatchNorm or block_type == LayerNorm:
                block = block_type(dim) 
            elif block_type == GPSConv:
                nn = Sequential(
                    Linear(dim, dim),
                    ReLU(),
                    Linear(dim, dim),
                )
                block = GPSConv(dim, GINEConv(nn), heads=4, attn_dropout=0.5)
            else:
                block = block_type()
            blocks2.append(block)
        else:
            raise ValueError(f"Component '{component}' not found in architecture dictionary.")
    
    model = NasGNN(blocks1 + [pooling], blocks2 + [pooling2], aggrs, aggrs2, aggr_type).to(device)

    return model