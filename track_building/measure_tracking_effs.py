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

model_dir = '../trained_models/for_paper'
models = os.listdir(model_dir)
model_paths = [model for model in models if 'epoch250' in model]
models_by_pt = {model.split('.')[0].split('_')[-1].strip('GeV'): model for model in model_paths}

# initial discriminants (won't matter in the end)
thlds = {'2': 0.2185, '1p5': 0.17994, '1': 0.0505, '0p9': 0.0447, '0p8': 0.03797, '0p7': 0.02275, '0p6': 0.01779}
device = "cpu"

pt_min = sys.argv[1]
model = models_by_pt[pt_min]
print('...evaluating', model)
thld = thlds[pt_min]
interaction_network = InteractionNetwork(40)
interaction_network.load_state_dict(torch.load(os.path.join(model_dir,model),
                                               map_location=torch.device('cpu')))
interaction_network.eval()

construction = 'heptrkx_plus_pyg'
graph_indir = "../../hitgraphs_3/{}_{}/".format(construction, pt_min)
print('...sampling graphs from:', graph_indir)
graph_files = np.array(os.listdir(graph_indir))
graph_files = [os.path.join(graph_indir, f) for f in graph_files]
n_graphs = len(graph_files)

# create a test dataloader 
params = {'batch_size': 1, 'shuffle': True, 'num_workers': 6}
test_set = GraphDataset(graph_files=graph_files) 
test_loader = DataLoader(test_set, **params)

pt_bins = np.array([0.6, 0.8, 1, 1.2, 1.5, 1.8, 2.1, 2.5, 3,
                    3, 3.5, 4, 5, 6, 8, 10, 15, 25, 40, 60, 100, 150])
pt_bin_centers = (pt_bins[1:] + pt_bins[:-1])/2.
effs_by_pt = {'tight': [],
              'exa': [],
              'cms': []}

eta_bins = np.linspace(-4,4,28)
eta_bin_centers = (eta_bins[1:] + eta_bins[:-1])/2.
effs_by_eta = {'tight': [],
               'exa': [],
               'cms': []}

cms_effs, tight_effs, exa_effs = [], [], []
with torch.no_grad():
    counter = 0

    found_by_pt = {'tight': np.zeros(len(pt_bin_centers)),
                   'exa': np.zeros(len(pt_bin_centers)),
                   'cms': np.zeros(len(pt_bin_centers))}
    missed_by_pt = {'tight': np.zeros(len(pt_bin_centers)),
                    'exa': np.zeros(len(pt_bin_centers)),
                    'cms': np.zeros(len(pt_bin_centers))}
    found_by_eta = {'tight': np.zeros(len(eta_bin_centers)),
                    'exa': np.zeros(len(eta_bin_centers)),
                    'cms': np.zeros(len(eta_bin_centers))}
    missed_by_eta = {'tight': np.zeros(len(eta_bin_centers)),
                     'exa': np.zeros(len(eta_bin_centers)),
                     'cms': np.zeros(len(eta_bin_centers))}

    for batch_idx, data in enumerate(test_loader):
        data = data.to(device)
        output = interaction_network(data)
        cutoff = int(len(data.edge_index[1])/2.)
        y, out = data.y[:cutoff], torch.min(torch.stack([output[:cutoff], output[cutoff:]]), dim=0).values.squeeze()
        # data.y[:cutoff,], (output[:cutoff,0] + output[cutoff:,0])/2.
        
        TP = torch.sum((y==1) & (out>thld)).item()
        TN = torch.sum((y==0) & (out<thld)).item()
        FP = torch.sum((y==0) & (out>thld)).item()
        FN = torch.sum((y==1) & (out<thld)).item()
        acc = ((TP+TN)/(TP+TN+FP+FN))
        loss = F.binary_cross_entropy(output.squeeze(1), data.y,
                                      reduction='mean').item()
        
        # count hits per pid in each event, add indices to hits
        X, pids, idxs = data.x, data.pid, data.edge_index[:,:cutoff]
        pts, etas, = data.pt, data.eta
        unique_pids = torch.unique(pids)
        pid_counts_map = {p.item(): torch.sum(pids==p).item() for p in unique_pids}
        n_particles = np.sum([counts >= 1 for counts in pid_counts_map.values()])
        pid_label_map = {p.item(): -5 for p in unique_pids}
        pid_pt_map = {pids[i].item(): pts[i].item() for i in range(len(pids))}
        pid_eta_map = {pids[i].item(): etas[i].item() for i in range(len(etas))}
        pid_found_map = {'tight': {p.item(): False for p in unique_pids
                                   if pid_counts_map[p.item()] >= 1},
                         'exa': {p.item(): False for p in unique_pids
                                 if pid_counts_map[p.item()] >= 1},
                         'cms': {p.item(): False for p in unique_pids
                                 if pid_counts_map[p.item()] >= 1}}

        #print(np.histogram(list(pid_eta_map.values())))
        hit_idx = torch.unsqueeze(torch.arange(X.shape[0]), dim=1)
        X = torch.cat((hit_idx.float(), X), dim=1)
        
        # separate segments into incoming and outgoing hit positions 
        good_edges = (out>thld)
        idxs = idxs[:,good_edges]
        feats_o = X[idxs[0]]
        feats_i = X[idxs[1]]

        # geometric quantities --> distance calculation 
        r_o, phi_o, z_o = 1000*feats_o[:,1], np.pi*feats_o[:,2], 1000*feats_o[:,3]
        eta_o = calc_eta(r_o, z_o)
        r_i, phi_i, z_i = 1000*feats_i[:,1], np.pi*feats_i[:,2], 1000*feats_i[:,3]
        eta_i = calc_eta(r_i, z_i)
        dphi, deta = calc_dphi(phi_o, phi_i), eta_i-eta_o
        distances = torch.sqrt(dphi**2 + deta**2)
        dist_matrix = 100*torch.ones(X.shape[0], X.shape[0])
        for i in range(dist_matrix.shape[0]): dist_matrix[i][i]=0
            
        for h in range(len(feats_i)):
            dist_matrix[int(feats_o[h][0])][int(feats_i[h][0])] = distances[h]
            #dist_matrix[int(feats_i[h][0])][int(feats_o[h][0])] = distances[h]
            
        # run DBScan
        eps_dict = {'2': 0.4, '1p5': 0.4, '1': 0.4, '0p9': 0.4, '0p8': 0.4, '0p7': 0.4, '0p6': 0.4}
        eps, min_pts = eps_dict[pt_min], 1
        clustering = DBSCAN(eps=eps, min_samples=min_pts,
                            metric='precomputed').fit(dist_matrix)
        labels = clustering.labels_
        
        # count reconstructed particles from hit clusters 
        cms_clusters, tight_clusters, exa_clusters = 0, 0, 0
        for label in np.unique(labels):  
            if label<0: continue # ignore noise 
                
            # grab pids corresponding to hit cluster labels
            label_pids = pids[labels==label]
            selected_pid = np.bincount(label_pids).argmax() # most frequent pid in cluster
            if pid_counts_map[selected_pid] < 1: continue

            # fraction of hits with the most common pid 
            n_selected_pid = len(label_pids[label_pids==selected_pid])
            selected_pid_fraction = n_selected_pid/len(label_pids)
                
            #previously_found = pid_label_map[selected_pid] > -1
            pid_label_map[selected_pid] = label
            label_pt = pid_pt_map[selected_pid]
            
            if selected_pid_fraction > 0.75:
                cms_clusters += 1 # all hits have the same pid
                pid_found_map['cms'][selected_pid] = True

                if pid_counts_map[selected_pid] == len(label_pids):
                    tight_clusters += 1 # all required hits for pid
                    pid_found_map['tight'][selected_pid] = True

            if selected_pid_fraction > 0.5:
                true_counts = pid_counts_map[selected_pid]
                if n_selected_pid/true_counts > 0.5:
                    exa_clusters += 1
                    pid_found_map['exa'][selected_pid] = True

        for d, pid_found in pid_found_map.items():
            for key, val in pid_found.items():
                pid_pt = pid_pt_map[key]
                pid_eta = pid_eta_map[key]
                for j in range(len(pt_bins)-1):
                    if (pid_pt < pt_bins[j+1]) and (pid_pt > pt_bins[j]):
                        if val==True: 
                            found_by_pt[d][j]+=1
                        else:
                            missed_by_pt[d][j]+=1
                for j in range(len(eta_bins)-1):
                    if (pid_eta < eta_bins[j+1]) and (pid_eta > eta_bins[j]):
                        if val==True:
                            found_by_eta[d][j]+=1
                        else:
                            missed_by_eta[d][j]+=1

            eff_by_pt = found_by_pt[d]/(found_by_pt[d]+missed_by_pt[d])
            eff_by_eta = found_by_eta[d]/(found_by_eta[d]+missed_by_eta[d])
            print(d, np.sum(found_by_pt[d])/(np.sum(found_by_pt[d])+np.sum(missed_by_pt[d])))
            print(d, np.sum(found_by_eta[d])/(np.sum(found_by_eta[d])+np.sum(missed_by_eta[d])))
            effs_by_pt[d].append(eff_by_pt)
            effs_by_eta[d].append(eff_by_eta)

        cms_effs.append(cms_clusters/n_particles)
        tight_effs.append(tight_clusters/n_particles)
        exa_effs.append(exa_clusters/n_particles)
        
        counter += 1
        #if (counter > 5): break

print("CMS Eff: {:.4f}+/-{:4f}".format(np.mean(cms_effs), np.std(cms_effs)))
print("Tight Eff: {:.4f}+/-{:.4f}".format(np.mean(tight_effs), np.std(tight_effs)))
print("Exa Eff: {:.4f}+/-{:.4f}".format(np.mean(exa_effs), np.std(exa_effs)))

pt_output = {}
for d, effs in effs_by_pt.items():
    print("Clustering Method:", d)
    effs = np.array(effs)
    mean_effs = np.nanmean(effs, axis=0)
    std_effs = np.nanstd(effs, axis=0)
    for i in range(len(pt_bin_centers)):
        print(pt_bin_centers[i], ' GeV: ', 
              mean_effs[i], ' +- ', 
              std_effs[i])

    pt_output[d] = {'bins': pt_bins,
                    'bin_centers': pt_bin_centers,
                    'effs_mean': mean_effs,
                    'effs_std': std_effs}
        

eta_output = {}
for d, effs in effs_by_eta.items():
    print("Clustering Method:", d)
    effs = np.array(effs)
    mean_effs = np.nanmean(effs, axis=0)
    std_effs = np.nanstd(effs, axis=0)
    for i in range(len(eta_bin_centers)):
        print(eta_bin_centers[i], ' GeV: ',
              mean_effs[i], ' +- ',
              std_effs[i])

    eta_output[d] = {'bins': eta_bins,
                     'bin_centers': eta_bin_centers,
                     'effs_mean': mean_effs,
                     'effs_std': std_effs}

output = {'pt': pt_output, 'eta': eta_output}
np.save('efficiencies/train3_{}GeV'.format(pt_min), output)
