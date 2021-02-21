import os
import sys
import random
import time
import argparse
sys.path.append('../')

import pickle
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.cluster import DBSCAN

from models.dataset import Dataset
from models.interaction_network import InteractionNetwork
from models.graph import Graph, save_graphs, load_graph


def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2*np.pi
    dphi[dphi < -np.pi] += 2*np.pi
    return dphi

def calc_eta(r, z):
    theta = np.arctan2(r, z)
    return -1. * np.log(np.tan(theta / 2.))

pt_cut = 1
use_cuda = False
construction = 'heptrkx_plus'
epoch = 20
disc = 0.299
model = "../{}_epoch{}_{}GeV.pt".format(construction, epoch, pt_cut)
print("model={0}".format(model))

# load in test graph paths
graph_indir = "../../hitgraphs/{}_{}/".format(construction, pt_cut)
graph_files = np.array(os.listdir(graph_indir))

device = torch.device("cuda" if use_cuda else "cpu")

train_kwargs = {'batch_size': 1}
test_kwargs = {'batch_size': 1}

n_graphs = len(graph_files)
IDs = np.arange(n_graphs)
#np.random.shuffle(IDs)
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

interaction_network = InteractionNetwork(3, 4, 4)
interaction_network.load_state_dict(torch.load(model, map_location=torch.device('cpu')))
interaction_network.eval()    

good_eff, tight_eff = [], []
with torch.no_grad():
    for data, target in test_loader:
        X, Ra = data['X'].float().to(device), data['Ra'].float().to(device)
        Ri, Ro = data['Ri'].float().to(device), data['Ro'].float().to(device)
        pids = target['pid'][0].to(device)
        print(pids.shape)
        target = target['y'].to(device)
        output = interaction_network(X, Ra.float(), Ri.float(), Ro.float())
        N_correct = torch.sum((target==1).squeeze() & (output>0.5).squeeze())
        N_correct += torch.sum((target==0).squeeze() & (output<0.5).squeeze())
        N_total = target.shape[1]
        accuracy = torch.sum(((target==1).squeeze() &
                              (output>disc).squeeze()) |
                             ((target==0).squeeze() &
                              (output<disc).squeeze())).float()/target.shape[1]
        test_loss = F.binary_cross_entropy(output.squeeze(2), target,
                                           reduction='mean').item()
        print('accuracy={}, loss={}'.format(accuracy, test_loss))

        true_edges = (output>disc).squeeze()
        hit_idx = torch.unsqueeze(torch.arange(X[0].shape[1]), dim=0)
        
        print('pids:', len(np.unique(pids)))
        pid_counts = {p.item(): torch.sum(pids==p).item() for p in pids}

        X = torch.cat((hit_idx.float(), X[0]), dim=0)
        
        feats_o = torch.matmul(Ro[0], torch.transpose(X, dim0=0, dim1=1))
        feats_o = feats_o[true_edges]
        feats_i = torch.matmul(Ri[0], torch.transpose(X, dim0=0, dim1=1))
        feats_i = feats_i[true_edges]

        r_o, phi_o, z_o = feats_o[:,1], feats_o[:,2], feats_o[:,3]
        eta_o = calc_eta(r_o, z_o)
        r_i, phi_i, z_i = feats_i[:,1], feats_i[:,2], feats_i[:,3]
        eta_i = calc_eta(r_i, z_i)
        dphi, deta = calc_dphi(phi_o, phi_i), eta_i-eta_o

        distances = torch.sqrt((r_i*torch.cos(np.pi*phi_i) - r_o*torch.cos(np.pi*phi_o))**2 + 
                               (r_i*torch.sin(np.pi*phi_i) - r_i*torch.sin(np.pi*phi_i))**2 + 
                               (z_i-z_o)**2)
        #distances = torch.sqrt(dphi**2 + deta**2)
 
        dist_matrix = 10*torch.ones(X.shape[1], X.shape[1])
        for h in range(len(feats_i)):
            dist_matrix[int(feats_o[h][0])][int(feats_i[h][0])] = distances[h]
     
        # DBScan
        min_pts = 1
        n_particles = len(np.unique(pids))
        for eps in [0.38]:
        #[0.05, 0.07, 0.08,  0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.24, 0.28]:
            clustering = DBSCAN(eps=eps, min_samples=min_pts,
                                metric='precomputed').fit(dist_matrix)
            labels = clustering.labels_
            print('eps={}, unique labels={}'
                  .format(eps, np.unique(labels[labels>-1]).shape))
            good_clusters = 0
            tight_clusters = 0
            for label in np.unique(labels):
                label_pids = pids[labels==label]
                if len(np.unique(label_pids)) == 1: 
                    good_clusters+=1
                    pid = label_pids[0].item()
                    if pid_counts[pid] == len(label_pids):
                        tight_clusters += 1

            good_eff.append(good_clusters/n_particles)
            tight_eff.append(tight_clusters/n_particles)
            print("GOOD: {}/{}={}".format(good_clusters, n_particles, good_clusters/n_particles)) 
            print("TIGHT: {}/{}={}".format(tight_clusters, n_particles, tight_clusters/n_particles))

    test_loss /= len(test_loader.dataset)
    accuracy /= len(test_loader.dataset)
    
print("Good Eff: {}+/-{}", np.mean(good_eff), np.std(good_eff))
print("Tight Eff: {}+/-{}", np.mean(tight_eff), np.std(tight_eff))
