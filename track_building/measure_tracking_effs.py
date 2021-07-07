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

def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2*np.pi
    dphi[dphi < -np.pi] += 2*np.pi
    return dphi

def calc_eta(r, z):
    theta = np.arctan2(r, z)
    return -1. * np.log(np.tan(theta / 2.))

model_dir = '../trained_models'
models = os.listdir(model_dir)
model_paths = [model for model in models if 'epoch250' in model]
models_by_pt = {model.split('.')[0].split('_')[-1].strip('GeV'): model for model in model_paths}
print(models_by_pt)

# initial discriminants (won't matter in the end)
thlds = {'2': 0.2185, '1p5': 0.1657, '1': 0.0505, '0p9': 0.0447, '0p8': 0.03797, '0p7': 0.02275, '0p6': 0.01779}
device = "cpu"

pt = sys.argv[1]
model = models_by_pt[pt]
print('...evaluating', model)
thld = thlds[pt]
interaction_network = InteractionNetwork(40)
interaction_network.load_state_dict(torch.load(os.path.join(model_dir,model),
                                               map_location=torch.device('cpu')))
interaction_network.eval()

construction = 'heptrkx_plus_pyg'
graph_indir = "../../hitgraphs_3/{}_{}/".format(construction, pt)
print('...sampling graphs from:', graph_indir)
graph_files = np.array(os.listdir(graph_indir))
graph_files = [os.path.join(graph_indir, f) for f in graph_files]
n_graphs = len(graph_files)

# create a test dataloader 
params = {'batch_size': 1, 'shuffle': True, 'num_workers': 6}
test_set = GraphDataset(graph_files=graph_files) 
test_loader = DataLoader(test_set, **params)

good_per_pt = []
tight_per_pt = []
exa_per_pt = []
good_effs, tight_effs, exa_effs = [], [], []
with torch.no_grad():
    counter = 0
    for batch_idx, data in enumerate(test_loader):
        data = data.to(device)
        output = interaction_network(data)
        cutoff = int(len(data.edge_index[1])/2.)
        y, out = data.y[:cutoff], torch.min(torch.stack([output[:cutoff], output[cutoff:]]), dim=0).values.squeeze()
        # data.y[:cutoff,], (output[:cutoff,0] + output[cutoff:,0])/2.
        
        TP = torch.sum((y==1) & (out>thld))#.item()
        TN = torch.sum((y==0) & (out<thld)).item()
        FP = torch.sum((y==0) & (out>thld)).item()
        FN = torch.sum((y==1) & (out<thld)).item()
        acc = ((TP+TN)/(TP+TN+FP+FN)).item()
        loss = F.binary_cross_entropy(output.squeeze(1), data.y,
                                      reduction='mean').item()
        
        # count hits per pid in each event, add indices to hits
        X, pids, idxs, pts = data.x, data.pid, data.edge_index[:,:cutoff], data.pt
        print(pts)
        n_particles = len(np.unique(pids))
        pid_counts = {p.item(): torch.sum(pids==p).item() for p in pids}
        pid_label_map = {p.item(): -5 for p in pids}
        hit_idx = torch.unsqueeze(torch.arange(X.shape[0]), dim=1)
        X = torch.cat((hit_idx.float(), X), dim=1)
        
        # separate segments into incoming and outgoing hit positions 
        feats_o = X[idxs[0]][out>thld]
        feats_i = X[idxs[1]][out>thld]
        
        # geometric quantities --> distance calculation 
        r_o, phi_o, z_o = feats_o[:,1], feats_o[:,2], feats_o[:,3]
        r_i, phi_i, z_i = feats_i[:,1], feats_i[:,2], feats_i[:,3]
        distances = torch.sqrt((r_i*torch.cos(np.pi*phi_i) - r_o*torch.cos(np.pi*phi_o))**2 +
                               (r_i*torch.sin(np.pi*phi_i) - r_i*torch.sin(np.pi*phi_i))**2 +
                               (z_i-z_o)**2)
        
        dist_matrix = 100*torch.ones(X.shape[0], X.shape[0])
        for i in range(dist_matrix.shape[0]): dist_matrix[i][i]=0
            
        for h in range(len(feats_i)):
            dist_matrix[int(feats_o[h][0])][int(feats_i[h][0])] = distances[h]
            dist_matrix[int(feats_i[h][0])][int(feats_o[h][0])] = distances[h]
            
        # run DBScan
        #eps_dict = {'2': 0.4, '1p5': 0.4, '1': 0.4, '0p9': 0.4, '0p8': 0.5, '0p7': 0.5, '0p6': 0.5}
        eps, min_pts = 0.5, 1
        clustering = DBSCAN(eps=eps, min_samples=min_pts,
                            metric='precomputed').fit(dist_matrix)
        labels = clustering.labels_
        
        # count reconstructed particles from hit clusters 
        good_clusters, tight_clusters, exa_clusters = 0, 0, 0
        for label in np.unique(labels):  
            if label<0: continue # ignore noise 
                
            # grab pids corresponding to hit cluster labels
            label_pids = pids[labels==label]
            selected_pid = np.bincount(label_pids).argmax() # most frequent pid in cluster
            
            # fraction of hits with the most common pid 
            n_reco_selected = len(label_pids[label_pids==selected_pid])
            hit_fraction = n_reco_selected/len(label_pids)
                
            #previously_found = pid_label_map[selected_pid] > -1
            pid_label_map[selected_pid] = label
            if hit_fraction > 0.99:
                good_clusters += 1
                #if not previously_found: good_clusters+=1
                pid = label_pids[0].item()
                if pid_counts[pid] == len(label_pids):
                    tight_clusters += 1
            if hit_fraction > 0.5:
                true_counts = pid_counts[selected_pid]
                if n_reco_selected/true_counts > 0.5:
                    exa_clusters += 1

        good_effs.append(good_clusters/(len(np.unique(labels))-1))
        tight_effs.append(tight_clusters/n_particles)
        exa_effs.append(exa_clusters/n_particles)
        counter += 1
        if (counter%20==0): print(f'...checkpoint {counter}')
        if (counter > 80): break

print("Good Eff: {:.4f}+/-{:4f}".format(np.mean(good_effs), np.std(good_effs)))
print("Tight Eff: {:.4f}+/-{:.4f}".format(np.mean(tight_effs), np.std(tight_effs)))
print("Exa Eff: {:.4f}+/-{:.4f}".format(np.mean(exa_effs), np.std(exa_effs)))
