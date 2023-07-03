import torch
import os
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

import tqdm
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logZ = torch.zeros((1,)).to(device)

n_hid = 256

# transformer params
n_heads = 4
n_layers = 2

mlp = make_mlp([params.emb_dim] + [n_hid] * n_layers + [params.n_actions]).to(device)
d_model = mlp[0].in_features
encoder_layer = nn.TransformerEncoderLayer(d_model=100, nhead=4).to(device)
transformer_layer = nn.TransformerEncoder(encoder_layer, num_layers=4).to(device)
model = TransformerModel(params, transformer_layer, mlp).to(device)

########## Load Checkpoint here #############
#it = 65
#model = model.load_state_dict(torch.load(f'/drive/My Drive//controller-it{it}.pth'))

P_B = 1 # DAG & autoregressive sequence generation => tree

optim = torch.optim.Adam([ {'params':model.parameters(), 'lr':0.001}, {'params':[logZ], 'lr':0.01} ])
logZ.requires_grad_()

losses_TB = []
zs_TB = []
rewards_TB = []
all_visited_TB = []
l1log_TB = []

batch_size = 10 # training child GNNs is expensive
max_len = 30

n_train_steps = 100
R_evals = []
all_valid_archs_results = [] # this is where we'll eventually gather promising architectures from
total_invalid_archs_per_ep = []

for it in tqdm.trange(n_train_steps): 

    Z = logZ.exp()

    flag = True
    if flag:
        ll_diff = torch.zeros((batch_size,)).to(device)
        ll_diff += logZ
    else:
        in_probs = torch.ones(batch_size, dtype=torch.float, requires_grad=True).to(device)

    # Generate a tensor of expected sequence length
    generated = torch.LongTensor(batch_size, max_len)
    generated.fill_(-1)
    gen_len = torch.LongTensor(batch_size,).fill_(0) 
    unfinished_sents = gen_len.clone().fill_(1) 

    cur_len = 0
    while cur_len < max_len:
        
        state = generated[:,:cur_len] + 0
        tensor = model(state.to(device), lengths=gen_len.to(device))
        scores = tensor.sum(dim=1)
        scores = get_action_space(cur_len, scores)
        scores = scores.log_softmax(1)
        sample_temperature = 1
        probs = F.softmax(scores / sample_temperature, dim=1)
        assert torch.allclose(torch.sum(probs,1).to(device), torch.ones([batch_size]).to(device)) # sanity check to ensure normalization of probabilities
        next_action = torch.multinomial(probs, 1).squeeze(1)

        # update generations
        generated[:,cur_len] = next_action.cpu() * unfinished_sents + params.pad_index * (1 - unfinished_sents)
        gen_len.add_(unfinished_sents) 
        cur_len += 1

        # Trajectory Balance loss
        if flag:
            ll_diff += scores.gather(1, next_action.unsqueeze(-1)).squeeze(1)
        else:
            sample_in_probs = probs.gather(1, next_action.unsqueeze(-1)).squeeze(1)
            sample_in_probs[unfinished_sents == 0] = 1.
            in_probs = in_probs * sample_in_probs
      
        # stop when all architectures have a final dense layer (maps to predicted energies)
        if unfinished_sents.max() == 0:
            break

    generated_list = generated.tolist()

    invalid_archs_ctr = 0
    R = tuple()
    for generated in generated_list:
      try:
        generated = [str(item) for item in generated]
        print("generated: ", generated)
        lr = float(list(params.keys())[list(params.values()).index(int(generated[0]))])
        batch_size_gnn = int(list(params.keys())[list(params.values()).index(int(generated[1]))])

        epochs = 100
        #epochs = 2
        gnn_model = get_model(generated, device)
        optimizer = torch.optim.Adam(gnn_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                              factor=0.7, patience=5,
                                                              min_lr=0.00001)
        best_val_error, losses = train_architecture(gnn_model, batch_size_gnn, epochs)
        reward = torch.Tensor([1.0 - best_val_error])
        R += (reward,) 
        arch_results = (generated, best_val_error, losses, reward)
        all_valid_archs_results.append(arch_results)

      except: # invalid architecture
        #print("Invalid: ", generated)
        invalid_archs_ctr += 1
        reward = torch.Tensor([0.0])
        R += (reward,) 

    path = '/drive/My Drive/all_valid_archs_results.txt'  
    with open(path, 'w') as file:
        file.write(str(all_valid_archs_results) + '\n')
        file.close()
    
    print(f"{invalid_archs_ctr} total invalid architectures found in iter {it}")
    total_invalid_archs_per_ep.append(invalid_archs_ctr)

    R = torch.cat(R).to(device)
    assert R.is_contiguous() 
    # sanity check to ensure we get rewards for all child GNNs
    assert R.size()[0] == batch_size

    optim.zero_grad()
    if flag :
        ll_diff -= R
        loss = (ll_diff**2).sum()/batch_size
    else :
        loss = ((Z*in_probs / R).log()**2).sum()/batch_size

    loss.backward()
    optim.step()

    losses_TB.append(loss.item())
    zs_TB.append(Z.item())
    rewards_TB.append(R.mean().cpu())
    all_visited_TB.extend(generated)
    R_evals.append(R.cpu())

    torch.save(model.state_dict(), f'/drive/My Drive/controller-it{it}.pth')

    print('\nloss =', np.array(losses_TB[-100:]).mean(), 'Z =', Z.item(), "R =", np.array(rewards_TB[-100:]).mean() )
    if losses_TB[it] < 1.0: 
      break