#!/bin/bash
PT="0GeV75"
BATCHSIZE=1
GRAPHBATCHNUM=0
CONSTRUCTION="heptrkx_plus"
CUDA=1
SETUP="import os
os.environ['CUDA_VISIBLE_DEVICES']='0' # pick gpu
import torch
torch.set_grad_enabled(False)
from torchvision import datasets, transforms
import numpy as np
from models.dataset import Dataset
from models.interaction_network import InteractionNetwork
use_cuda = $CUDA
torch.manual_seed(1)
device = torch.device('cuda' if use_cuda else 'cpu')
construction = '$CONSTRUCTION'
pt = '$PT'
graph_indir = 'hitgraphs/{}_{}/'.format(construction, pt)
graph_files = np.array(os.listdir(graph_indir))
n_graphs = len(graph_files)
IDs = np.arange(n_graphs)
partition = {'test':  graph_files[IDs[:100]]}
params = {'batch_size': $BATCHSIZE, 'shuffle': False, 'num_workers': 0}
test_set = Dataset(graph_indir, partition['test'])
test_loader = torch.utils.data.DataLoader(test_set, **params)
model = InteractionNetwork(3, 4, 4).to(device)
model.eval()
i = 0
for data, target in test_loader:
  if i == $GRAPHBATCHNUM: break
  i += 1
X, Ra = data['X'].to(device), data['Ra'].to(device, dtype=torch.float32)
Ri, Ro = data['Ri'].to(device, dtype=torch.float32), data['Ro'].to(device, dtype=torch.float32)"
echo $SETUP
python -m timeit -s "$SETUP" -n 100 -r 2 -v "model(X, Ra, Ri, Ro)"
