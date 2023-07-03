import torch.nn as nn
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


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    """
    State Embeddings
    """
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m

def get_padding_masks(slen, lengths):
    """
    Generate hidden states mask
    """
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]
    assert mask.size() == (bs, slen)
    return mask


def make_mlp(l, act=nn.LeakyReLU(), tail=[]):
    """makes an MLP with no top layer activation"""
    return nn.Sequential(*(sum(
        [[nn.Linear(i, o)] + ([act] if n < len(l)-2 else [])
         for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))

class TransformerModel(nn.Module):

    def __init__(self, params, transformer_layer, linear_layer):
        """
        Controller
        """
        super().__init__()
        self.embeddings = Embedding(params.n_actions, params.emb_dim, padding_idx=params.pad_index)
        self.linear = linear_layer
        self.transformer_layer = transformer_layer

    def forward(self, x, lengths):
        """
        Inputs:
            `x` LongTensor(bs, slen), containing indices
            `lengths` LongTensor(bs), containing the length of each seq
        """
        
        # check inputs
        bs, slen = x.size()
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen

        # generate masks
        mask = get_padding_masks(slen, lengths)

        # embeddings
        tensor = self.embeddings(x)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)
        
        # transformer layer  
        tensor = self.transformer_layer(tensor)
        tensor = self.linear(tensor)
        
        return tensor