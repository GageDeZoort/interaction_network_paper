#!/bin/bash
PT="0GeV5"
BATCHSIZE=1
GRAPHBATCHNUM=4
CONSTRUCTION="heptrkx_plus_pyg"
CUDA=1
NUMBER=100
REPEAT=5
SETUP="import os
os.environ['CUDA_VISIBLE_DEVICES']='0' # pick gpu
import torch
torch.set_grad_enabled(False)
from torchvision import datasets, transforms
import numpy as np
from models.dataset import GraphDataset
from models.interaction_network import InteractionNetwork
use_cuda = $CUDA
torch.manual_seed(1)
device = torch.device('cuda' if use_cuda else 'cpu')
construction = '$CONSTRUCTION'
pt = '$PT'
graph_indir = 'hitgraphs/{}_{}/'.format(construction, pt)
graph_files = np.array(os.listdir(graph_indir))
graph_files = np.array([os.path.join(graph_indir, graph_file)
                        for graph_file in graph_files])
n_graphs = len(graph_files)
IDs = np.arange(n_graphs)
partition = {'test':  graph_files[IDs[:100]]}
params = {'batch_size': $BATCHSIZE, 'shuffle': False, 'num_workers': 0}
test_set = GraphDataset(graph_files=partition['test'])
data = test_set[$GRAPHBATCHNUM]
data = data.to(device)
model = InteractionNetwork(hidden_size=40).to(device)
model = model.jittable()
model.eval()
model = torch.jit.script(model)"
echo $SETUP
python3 -m timeit -s "$SETUP" -n "$NUMBER" -r "$REPEAT" -v "model(data.x, data.edge_index, data.edge_attr)"
