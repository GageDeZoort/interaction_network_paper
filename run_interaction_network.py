import os 

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
from models.dataset import Dataset


from models.PyG.interaction_network import InteractionNetwork
from models.PyG.dataset import Dataset

def train(args, model, device, train_loader, optimizer, epoch):
    #model.train()
    #epoch_t0 = time()
    #losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        x = data['x'].to(device), 
        edge_attr = data['edge_attr'].to(device)
        edge_index = data['edge_index'].to(device)
        target = target['y'].to(device)
        print('x', x)
        print('edge_attr', edge_attr)
        print('edge_index', edge_index)
        #optimizer.zero_grad()
        #output = model(X, Ra.float(), Ri.float(), Ro.float())
        #loss = F.binary_cross_entropy(output.squeeze(2), target)
        #loss.backward()
        #optimizer.step()
        #if batch_idx % args.log_interval == 0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, batch_idx * len(X), len(train_loader.dataset),
        #        100. * batch_idx / len(train_loader), loss.item()))
        #    if args.dry_run:
        #        break
        #losses.append(loss.item())
    print("...epoch time: {0}s".format(time()-epoch_t0))
    #print("...epoch {}: train loss={}".format(epoch, np.mean(losses)))

pt = 1
construction = "heptrkx_plus"
home_dir = "/scratch/gpfs/jdezoort"
graph_indir = "{}/hitgraphs/{}_{}/".format(home_dir, construction, pt)

graph_files = np.array(os.listdir(graph_indir))
n_graphs = len(graph_files)

IDs = np.arange(n_graphs)
np.random.shuffle(IDs)
partition = {'train': graph_files[IDs[:1000]],
             'test':  graph_files[IDs[1000:1400]],
             'val': graph_files[IDs[1400:1500]]}

params = {'batch_size': 1, 'shuffle': True, 'num_workers': 6}
train_set = Dataset(graph_indir, partition['train'])
train_loader = torch.utils.data.DataLoader(train_set, **params)
test_set = Dataset(graph_indir, partition['test'])
test_loader = torch.utils.data.DataLoader(test_set, **params)
val_set = Dataset(graph_indir, partition['val'])
val_loader = torch.utils.data.DataLoader(val_set, **params)

train(0, 0, 'cpu', train_loader, 0, 0)
