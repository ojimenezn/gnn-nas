# Neural Architecture Search for GNN Potentials with Reinforcement Learning
### Omar Jiménez, Avi Arora, Sasan Zohreh
##### CSE8803 – Machine Learning for Chemistry
#


**Hypothesis**: Performance of Graph Neural Networks in chemistry could potentially be improved by using NAS-based techniques

**Goal**: Reinforcement Learning-based NAS for materials datasets

# NAS Formulation
<img width="1073" alt="Screen Shot 2023-04-27 at 3 57 36 PM" src="https://user-images.githubusercontent.com/25010271/234977606-79b6c60b-05ea-43a6-9d10-564d057ffda5.png">

# Search Space
- **GC Layers**: CGConv, GATConv, GraphConv, GCNConv, Dense (tried others like Transformer/GPS Layers but those are difficult for single GPU)
- **Aggregation Operators**: Add, Mean, Max
- **Activation Functions**: ReLU, Sigmoid, Tanh, LeakyReLU, SiLU
- **Pooling Layers**: Global add/mean/max
- **Normalization Layers**: BatchNorm, LayerNorm, MessageNorm, GraphSizeNorm
- **Hyperparameters**: learning rate, batch size, GC/Dense layer dims




# Future Work

- Enhancement and Flexibility of design space (e.g., layer diversity). Ideally, if we can afford to train for extended periods, we could let controller learn how to build architectures from scratch. This is a typical approach in RL NAS 
- Parameter-sharing RL NAS: original plan due to more efficiency but trickier to implement, safer to start with more intuitive sequential approach
- Smaller datasets (e.g., Jarvis DFT) which could allow us to train each child GNN until convergence using sequential NAS formulation
- Training/Evaluation on full MP Dataset for best NAS architecture and baselines
- Less exploratory objectives if training for shorter times (e.g., simple REINFORCE loss)
- Molecule datasets since only Evolutionary Algorithms have been applied
- Force training for child GNNs (only energy in experiments) 



