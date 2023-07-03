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


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def convert_list_to_dict(lst):
    result_dict = {}
    for i in range(len(lst)):
        result_dict[lst[i]] = i
    return result_dict

def get_non_idxs(action_space):
  non_idxs = []
  for key, value in params.items():
    if key != 'n_actions' and key != 'emb_dim':
      if key not in action_space:
          non_idxs.append(value)
  return non_idxs

lr_space = ['0.005', '0.007']
batch_size_space = ['16', '32']
dims_space = ['64', '128']

gnn_layers_space = ['cgconv', 'graphconv', 'gatconv', 'gcnconv']
aggr_space = ['add', 'mean', 'max']
normalization_space = ['batch_norm', 'graph_size_norm', 'message_norm', 'layer_norm']
linear_layers_space = ['linear']
activation_space = ['relu', 'sigmoid', 'silu', 'tanh', 'leakyrelu']
pooling_space = ['global_add', 'global_mean', 'global_max']
skip_space = ['skip1', 'skip2', 'skip3']
block_space = ['concat', 'sum']

total_space = lr_space + batch_size_space + dims_space + gnn_layers_space + aggr_space + normalization_space + \
              linear_layers_space + activation_space + pooling_space + skip_space + block_space + [str(",")] #pad
space_size = len(total_space)
pad_idx = space_size - 1

params_dict = convert_list_to_dict(total_space)
params = AttrDict({"n_actions":space_size} | params_dict | {"pad_index":pad_idx} | {"emb_dim":100})

non_lr_idxs = get_non_idxs(lr_space)
non_batch_size_idxs = get_non_idxs(batch_size_space)
non_dims_idxs = get_non_idxs(dims_space)
non_gnn_layers_idxs = get_non_idxs(gnn_layers_space)
non_aggr_idxs = get_non_idxs(aggr_space)
non_normalization_idxs = get_non_idxs(normalization_space)
non_linear_layers_idxs = get_non_idxs(linear_layers_space)
non_activation_idxs = get_non_idxs(activation_space)
non_pooling_idxs = get_non_idxs(pooling_space)
non_skip_idxs = get_non_idxs(skip_space)
non_block_idxs = get_non_idxs(block_space)

import torch_geometric
architecture_dict = AttrDict({
    str(list(params.keys()).index('linear') - 1) : nn.Linear,
    str(list(params.keys()).index('graphconv') - 1): torch_geometric.nn.GraphConv,
    str(list(params.keys()).index('gatconv') - 1): torch_geometric.nn.GATConv,
    str(list(params.keys()).index('cgconv') - 1): torch_geometric.nn.CGConv,
    str(list(params.keys()).index('gcnconv') - 1): torch_geometric.nn.GCNConv,
    #str(list(params.keys()).index('gpsconv') - 1): torch_geometric.nn.GPSConv,
    str(list(params.keys()).index('batch_norm') - 1): torch_geometric.nn.BatchNorm,
    str(list(params.keys()).index('graph_size_norm') - 1): torch_geometric.nn.GraphSizeNorm,
    str(list(params.keys()).index('message_norm') - 1): torch_geometric.nn.MessageNorm,
    str(list(params.keys()).index('layer_norm') - 1): torch_geometric.nn.LayerNorm,
    str(list(params.keys()).index('relu') - 1): torch.nn.ReLU,
    str(list(params.keys()).index('tanh') - 1): torch.nn.Tanh,
    str(list(params.keys()).index('sigmoid') - 1): torch.nn.Sigmoid,
    str(list(params.keys()).index('leakyrelu') - 1): torch.nn.LeakyReLU,
    str(list(params.keys()).index('silu') - 1): torch.nn.SiLU,
    str(list(params.keys()).index('global_add') - 1): torch_geometric.nn.global_add_pool,
    str(list(params.keys()).index('global_max') - 1): torch_geometric.nn.global_max_pool,
    str(list(params.keys()).index('global_mean') - 1): torch_geometric.nn.global_mean_pool,
})

del params[',']
params