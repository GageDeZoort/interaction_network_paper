from __future__ import print_function
#import setGPU
import os
from time import time
import timeit
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from models.dataset import Dataset
from models.interaction_network import InteractionNetwork


parser = argparse.ArgumentParser(description='PyTorch Interaction Network Implementation')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=14, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--pt', type=str, default='2',
                    help='Cutoff pt value in GeV (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--construction', type=str, default='heptrkx_classic',
                    help='graph construction method')

args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
print("use_cuda={0}".format(use_cuda))

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

train_kwargs = {'batch_size': args.batch_size}
test_kwargs = {'batch_size': args.test_batch_size}
if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

graph_indir = "/interactionnetworkvol/interaction_network_paper/hitgraphs/{}_{}/".format(args.construction, args.pt)
graph_files = np.array(os.listdir(graph_indir))
n_graphs = len(graph_files)

IDs = np.arange(n_graphs)
np.random.shuffle(IDs)
partition = {'train': graph_files[IDs[:1]],
             'test':  graph_files[IDs[:100]],
             'val': graph_files[IDs[:3]]}

params = {'batch_size': 1, 'shuffle': True, 'num_workers': 0}
train_set = Dataset(graph_indir, partition['train'])
train_loader = torch.utils.data.DataLoader(train_set, **params)
test_set = Dataset(graph_indir, partition['test'])
test_loader = torch.utils.data.DataLoader(test_set, **params)
val_set = Dataset(graph_indir, partition['val'])
val_loader = torch.utils.data.DataLoader(val_set, **params)

model = InteractionNetwork(3, 4, 4).to(device)
#model = InteractionNetwork(3, 1, 1).to(device)
total_params = sum(p.numel() for p in model.parameters())
print("total params", total_params)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=5, gamma=args.gamma)

#device warmup
model.eval()
best_discs = []
for data, target in val_loader:
    X, Ra = data['X'].to(device), data['Ra'].to(device)
    Ri, Ro = data['Ri'].to(device), data['Ro'].to(device)
    #target = target['y'].to(device)
    target = target.to(device)
    output = model(X, Ra.float(), Ri.float(), Ro.float())
    N_correct = torch.sum((target==1).squeeze() & (output>0.5).squeeze())
    N_correct += torch.sum((target==0).squeeze() & (output<0.5).squeeze())
    N_total = target.shape[1]
    print("warming up...")

#inference timing measurements
acc = []

for epoch in range(1, args.epochs + 1):
    model.eval()
    test_loss = 0
    accuracy = 0
    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            X, Ra = data['X'].to(device), data['Ra'].to(device)
            Ri, Ro = data['Ri'].to(device), data['Ro'].to(device)
            #target = target['y'].to(device)
            target = target.to(device)
            output = model(X, Ra.float(), Ri.float(), Ro.float())
            temp = timeit.timeit('output', 'from __main__ import output')
            acc.append(temp)
            print(acc[count])
            if use_cuda:
                timings = open("gpu_timing.txt", "a")
            else:
                timings = open("cpu_timing.txt", "a")
            timings.write("{0}s \n".format(acc[count]))
            timings.close()
            count = count + 1

    scheduler.step()

avg = 0
for i in acc:
    avg += i

avg /= float(len(acc))
if use_cuda:
    timings = open("gpu_timing.txt", "a")
else:
    timings = open("cpu_timing.txt", "a")
timings.write("avg = {0}s \n".format(avg))
timings.close()
