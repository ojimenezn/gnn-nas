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

train_loader_32 = torch.load("/drive/My Drive/train_loader_32.pth")
test_loader_32 = torch.load("/drive/My Drive/test_loader_32.pth")
val_loader_32 = torch.load("/drive/My Drive/val_loader_32.pth")
train_loader_64 = torch.load("/drive/My Drive/train_loader_64.pth")
test_loader_64 = torch.load("/drive/My Drive/test_loader_64.pth")
val_loader_64 = torch.load("/drive/My Drive/val_loader_64.pth")

train_loader_32_15k = torch.load("/drive/My Drive/train_loader_32-15k.pth")
test_loader_32_15k = torch.load("/drive/My Drive/test_loader_32-15k.pth")
val_loader_32_15k = torch.load("/drive/My Drive/val_loader_32-15k.pth")
train_loader_64_15k = torch.load("/drive/My Drive/train_loader-64-15k.pth")
test_loader_64_15k = torch.load("/drive/My Drive/test_loader_64-15k.pth")
val_loader_64_15k = torch.load("/drive/My Drive/val_loader_64-15k.pth")

class ManualGNN(torch.nn.Module):
    """
    Manual GNN Constructor for debugging purposes
    """
    def __init__(self):

        super().__init__()
        self.lin0 = torch.nn.Linear(1, dim)

        self.bn = BatchNorm(dim)
        self.ln = LayerNorm(dim)

        ### Block 1 GC Layers
        self.conv1 = CGConv(dim, 1, aggr='add')
        self.conv2 = GATConv(dim, dim, aggr='mean')
        self.conv3 = CGConv(dim, 1, aggr='mean')

        ### Block 2 GC Layers (only 2, first one is another dense --> 14)
        self.lin3 = torch.nn.Linear(dim, dim)
        self.conv4 = GATConv(dim, dim, aggr='mean')
        self.conv5 = GraphConv(dim, dim, aggr='max')

        ################ Tests with Graph Transformer layers ##################
        
        # NOT PART OF NAS EXPERIMENTS. JUST TO SEE IF THEY COULD BE INCORPORATED 
        # TO SEARCH SPACE EVENTUALLY.

        #self.node_emb = Embedding(100, dim)
        #self.edge_emb = Embedding(8, dim)
        
        #nn = Sequential(
            #Linear(dim, dim),
            #ReLU(),
            #Linear(dim, dim),
        #)

        #self.conv2 = GPSConv(dim, GINEConv(nn), heads=1, attn_dropout=0.5)
        #self.conv2 = TransformerConv(dim, dim)

        #######################################################################

        self.lin1 = torch.nn.Linear(dim, dim)
        self.lin2 = torch.nn.Linear(dim*2, 1) # concat

    def forward(self, data):

        out1 = F.relu(self.lin0(data.x))

        out1 = F.relu(self.conv1(out1, data.edge_index, data.edge_attr))
        out1 = F.silu(self.conv2(out1, data.edge_index, data.edge_attr))
        out1 = self.ln(F.silu(self.conv3(out1, data.edge_index, data.edge_attr)))

        # Transformer/GPSConv layers give out memory errors 
        #x = self.node_emb(data.x.squeeze(-1).long()) 
        #edge_attr = self.edge_emb(data.edge_attr.long())
        #out = self.conv2(x, data.edge_index, data.batch, edge_attr=edge_attr)

        out1 = global_mean_pool(out1, data.batch)
        out1 = F.relu(self.lin1(out1))

        # BLOCK 2
        out2 = F.relu(self.lin0(data.x))
        out2 = self.ln(F.silu(self.conv4(out2, data.edge_index, data.edge_attr)))
        out2 = self.bn(self.conv5(out2, data.edge_index, data.edge_attr))
        out2 = global_max_pool(out2, data.batch)
        out2 = F.relu(self.lin1(out2))

        out = torch.cat((out1, out2), dim=1)
        out = self.lin2(out)

        return out.view(-1)


dim = 64 # 4 in sequence
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ManualGNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.7, patience=5,
                                                       min_lr=0.00001)

#Train and test functions
def train(model, train_loader, force_training):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        e = model(data)
        if force_training:
          f_pred = -1*torch.autograd.grad(
              outputs=e,
              inputs=data.pos,
              grad_outputs=torch.ones_like(e),
              create_graph=True, retain_graph = True,
          )[0]
          #loss = ef_loss(e, data.y, f_pred, data.force)
        else:
          loss = F.l1_loss(model(data), data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)

def test(model, test_loader, force_training):
    model.eval()
    loss_all = 0
    predicted_energies = []
    predicted_forces = []

    for data in test_loader:
        data = data.to(device)
        e = model(data)
        if force_training:
          f_pred = -1*torch.autograd.grad(
              outputs=e,
              inputs=data.pos,
              grad_outputs=torch.ones_like(e),
              create_graph=True, retain_graph = True,
          )[0]
          #loss = ef_loss(e, data.y, f_pred, data.force)
        else:
          loss = F.l1_loss(model(data), data.y)
        loss_all += loss.item() * data.num_graphs
        predicted_energies.append(model(data).cpu().detach().numpy())
    
    if force_training: return loss_all / len(test_loader.dataset), predicted_energies, predicted_forces
    else: return loss_all / len(test_loader.dataset), predicted_energies


def train_architecture(model, batch_size, epochs):

    #if batch_size == 32: train_loader, test_loader, val_loader = train_loader_32, test_loader_32, val_loader_32
    #else: train_loader, test_loader, val_loader = train_loader_64, test_loader_64, val_loader_64
  
    if batch_size == 32: train_loader, test_loader, val_loader = train_loader_32_15k, test_loader_32_15k, val_loader_32_15k
    else: train_loader, test_loader, val_loader = train_loader_64_15k, test_loader_64_15k, val_loader_64_15k

    best_val_error = None
    losses = []
    val_errors = []
    for epoch in range(1, epochs+1):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train(model, train_loader, force_training=False)
        val_error, _ = test(model, val_loader, force_training=False)
        scheduler.step(val_error)
        losses.append(loss)
        val_errors.append(val_error)

        if best_val_error is None or val_error <= best_val_error:
            test_error = test(model, test_loader, force_training=False)
            best_val_error = val_error

        print(f'Epoch: {epoch:03d}, LR: {lr:7f}, Loss: {loss:.7f},' f'Val MAE: {val_error:.7f}')
        
    return best_val_error, val_errors, losses

batch_size = 32 # 3 in sequence
epochs = 300
best_test_error, val_errors, losses = train_architecture(model, batch_size, epochs)
res = (best_test_error, val_errors, losses)
path = '/drive/My Drive/best-arch-log.txt'  
with open(path, 'w') as file:
    file.write(str(res))
    file.close()