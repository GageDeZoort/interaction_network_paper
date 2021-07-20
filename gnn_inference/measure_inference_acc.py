import os
import sys
sys.path.append('../')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from matplotlib import cm
from torch_geometric.data import Data, DataLoader

from models.interaction_network import InteractionNetwork
from models.graph import Graph
from models.dataset import GraphDataset

construction = sys.argv[1]
model_dir = '../trained_models/for_paper'
models = os.listdir(model_dir)
model_paths = [model for model in models if 'epoch250' in model]
models_by_pt = {model.split('.')[0].split('_')[-1].strip('GeV'): model for model in model_paths}
pts = np.array(list(models_by_pt.keys()))
models = np.array(list(models_by_pt.values()))
pts_sorted = pts.argsort()
pts, models = pts[pts_sorted], models[pts_sorted]
print("Sorted models", pts, "\n", models)
models_by_pt = {pts[i]: models[i] for i in range(len(pts))}
thlds = {'2': 0.2185, '1p5': 0.1657, '1': 0.0505, '0p9': 0.0447, '0p8': 0.03797, '0p7': 0.02275, '0p6': 0.01779}

use_cuda = torch.cuda.is_available()
print('...use_cuda=', use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")

n_pt = len(models)
output = {'losses': np.zeros((n_pt, n_pt)),
          'accs': np.zeros((n_pt, n_pt)),
          'loss_errs': np.zeros((n_pt, n_pt)),
          'acc_errs': np.zeros((n_pt, n_pt))}

for model_idx, (pt, model) in enumerate(models_by_pt.items()):
    print('...evaluating', model)
    thld = thlds[pt]

    interaction_network = InteractionNetwork(40).to(device)
    interaction_network.load_state_dict(torch.load(os.path.join(model_dir,model),
                                                   map_location=torch.device(device)))
    interaction_network.eval()

    for test_idx, test_pt in enumerate(models_by_pt.keys()):
        graph_indir = "../../hitgraphs_3/{}_{}/".format(construction, test_pt)
        print('...sampling graphs from:', graph_indir)
        graph_files = np.array(os.listdir(graph_indir))
        graph_files = [os.path.join(graph_indir, f) for f in graph_files]
        n_graphs = len(graph_files)

        # create a test dataloader 
        params = {'batch_size': 1, 'shuffle': True, 'num_workers': 6}
        test_set = GraphDataset(graph_files=graph_files) 
        test_loader = DataLoader(test_set, **params)
    
        accs, losses = [], []
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                data = data.to(device)
                out = interaction_network(data)
                y, out = data.y, out.squeeze()
            
                TP = torch.sum((y==1) & (out>thld)).item()
                TN = torch.sum((y==0) & (out<thld)).item()
                FP = torch.sum((y==0) & (out>thld)).item()
                FN = torch.sum((y==1) & (out<thld)).item()
                acc = ((TP+TN)/(TP+TN+FP+FN))
                loss = F.binary_cross_entropy(out, y, reduction='mean').item()

                accs.append(acc)
                losses.append(loss)

        test_idx_down = len(pts)-1-test_idx
        output['losses'][model_idx, test_idx_down] = np.mean(losses)
        output['loss_errs'][model_idx, test_idx_down] = np.std(losses)
        output['accs'][model_idx, test_idx_down] = np.mean(accs)
        output['acc_errs'][model_idx, test_idx_down] = np.std(accs)
        
        print('{}({}), {}({})'.format(pt, model_idx, test_pt, -test_idx))
        print(output['losses'][model_idx, test_idx_down], '+/-', 
              output['loss_errs'][model_idx, test_idx_down])
        print(output['accs'][model_idx, test_idx_down], '+/-',
              output['acc_errs'][model_idx, test_idx_down])

np.save('accuracies/{}_accuracies'.format(construction), output)
