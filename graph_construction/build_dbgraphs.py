#!/usr/bin/env

import os
import sys
import time
from os import listdir
from os.path import isfile, join
sys.path.append('../')

from numba import jit
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from matplotlib import rc

import trackml
from trackml.dataset import load_event
from trackml.dataset import load_dataset
from models.graph import Graph, save_graph
from models.track_fitter import TrackFitter

valid_layer_pairs = [(0,1), (1,2), (2,3),
                     (4,5), (5,6), (6,7), 
                     (7,8), (8,9), (9,10),
                     (11,12), (12,13), (13,14), 
                     (14,15), (15,16), (16,17),
                     (0,4), (1,4), (2,4), (3,4),
                     (0,11), (1,11), (2,11), (3,11)]

pixel_layers = [(8,2), (8,4), (8,6), (8,8),
                (7,14), (7,12), (7,10),
                (7,8), (7,6), (7,4), (7,2),
                (9,2), (9,4), (9,6), (9,8),
                (9,10), (9,12), (9,14)]


#########################  
## DBCLUSTER FUNCTIONS ##
#########################
@jit(nopython=True)
def build_dist_matrix(X):
    """ speedy custom distance matrix calculation
    """
    n_hits = X.shape[0]
    dist_matrix = 100.*np.ones((n_hits, n_hits))
    np.fill_diagonal(dist_matrix, 0.0)
    for i in range(n_hits):
        for j in range(i, n_hits):
            if (X[i][2] == X[j][2]): continue
            dEta = X[i][0] - X[j][0]
            dPhi = min(abs(X[i][1] - X[j][1]),
                       2*np.pi - abs(X[i][1]-X[j][1]))**2
            dist = np.sqrt(dEta**2 + dPhi**2)
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
    return dist_matrix
    
def cluster_hits(dist_matrix, eps=0.05, k=3):
    clustering = DBSCAN(eps=eps, min_samples=k,
                        metric='precomputed').fit(dist_matrix)
    return clustering.labels_

def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2*np.pi
    dphi[dphi < -np.pi] += 2*np.pi
    return dphi

def calc_eta(r, z):
    theta = np.arctan2(r, z)
    return -1. * np.log(np.tan(theta / 2.))

#########################
## GRAPH CONSTRUCTION ##
#########################
data_dir = '/tigress/jdezoort/train_1'
files = [f for f in listdir(data_dir)]
files = [f.split('.')[0] for f in files if "hits" in f]
evt_ids = [f.split('-')[0] for f in files]

pt_cut = 2
construction = 'LP'
db_params = {1: [0.11, 2], 2: [0.3, 3], 1.5: [0.08, 3], 5: [0.2, 3]}
pt_cut_tostr = {0: "0", 0.5: "0p5", 0.6: "0p6", 1: "1", 1.5: "1p5", 2: "2", 5: "5"}
outdir = "/scratch/gpfs/jdezoort/hitgraphs/dbgraphs_{}".format(pt_cut_tostr[pt_cut])

if (construction=='LPP'):
    valid_layer_pairs.extend([(0,0), (1,1), (2,2), (3,3), (4,4),
                              (5,5), (6,6), (7,7), (8,8), (9,9),
                              (10,10), (11,11), (12,12), (13,13),
                              (14,14), (15,15), (16,16), (17,17)])
    
for evt_id in evt_ids:
    #if (int(evt_id.split("00000")[1].split(".")[0]) > 1004): continue

    hits, cells, particles, truth = load_event(os.path.join(data_dir, evt_id))
    hits_by_loc = hits.groupby(['volume_id', 'layer_id'])
    hits = pd.concat([hits_by_loc.get_group(pixel_layers[i]).assign(layer=i)
                      for i in range(len(pixel_layers))])
    
    # outer merge keeps noise hits (with particle_id=0)
    truth = (truth[['hit_id', 'particle_id', 'weight']]
             .merge(particles[['particle_id']], on='particle_id'))
    
    # cylindrical r, detector phi and eta, conformal u and v
    r = np.sqrt(hits.x**2 + hits.y**2)
    eta = calc_eta(r, hits.z)
    phi = np.arctan2(hits.y, hits.x)
    
    hits = (hits[['hit_id', 'x', 'y', 'z', 'layer']]
            .assign(r=r, eta=eta, phi=phi)
            .merge(truth[['hit_id', 'particle_id']], on='hit_id'))
    
    # enforce the layer pairs graph construction
    if (construction == 'LP'):
        print('one particle per layer')
        hits = hits.loc[
            hits.groupby(['particle_id', 'layer'], as_index=False).r.idxmin()
        ]

    pt = np.sqrt(particles.px**2 + particles.py**2)
    particles['pt'] = pt

    # select hits and particles above the pt threshold
    particles_ptcut = np.unique(particles[pt > pt_cut].particle_id)
    print('num particles', len(particles_ptcut))
    hits = hits[hits['particle_id'].isin(particles_ptcut)]
    
    #####################
    ## CALCULATE TRUTH ##
    #####################
    fitter = TrackFitter()
    fitter.generate_truth(hits, particles, fit=False, 
                          construction=construction)
    n_truth_edges = fitter.n_truth_segs
    print('n_truth_edges', n_truth_edges)

    #####################
    ## EXTEND SEGMENTS ##
    #####################
    hits = hits.reset_index()
    segs_in, segs_out = np.array([]), np.array([])
    y = np.array([])
    dr_all, dphi_all, dz_all, dR_all = [], [], [], []
    X = hits[['eta', 'phi', 'r', 'z', 'layer']].to_numpy()
    dist_matrix = build_dist_matrix(X)
    params = db_params[pt_cut]
    labels = cluster_hits(dist_matrix, params[0], params[1])
    #print(labels)
    colors = ['b', 'y', 'g', 'r', 'pink', 'crimson', 'darkmagenta', 'rebeccapurple', 'midnightblue', 'slategrey', 'lightskyblue', 'mediumseagreen', 'gold', 'darkorange', 'darkred']
    #for label in labels:
    #    plt.scatter(hits[(np.array(labels)==label)]['eta'].values, 
    #                hits[(np.array(labels)==label)]['phi'].values,
    #                color=colors[label%4])
    #plt.show()

    for label in np.unique(labels):
        if (label < 0): continue # ignore noise
        
        cluster = hits[(np.array(labels) == label)]
        #print("cluster:", cluster)
        cluster_indices = cluster.index.to_numpy()
        #print("cluster_indices", cluster_indices)
        segments = cluster_indices[np.transpose(np.triu_indices(len(cluster_indices), 1))]
        #print("segments", segments)

        # ignore cross-layer hits
        pid1 = hits['particle_id'][segments[:,0]].values
        pid2 = hits['particle_id'][segments[:,1]].values
        #print("pid1", pid1)
        #print("pid2", pid2)

        lid1 = hits['layer'][segments[:,0]].values
        lid2 = hits['layer'][segments[:,1]].values
        next_layer_segs = np.array([i in valid_layer_pairs
                                    for i in list(zip(lid1, lid2))],
                                   dtype=bool)

        pid1, pid2 = pid1[next_layer_segs], pid2[next_layer_segs]
        segments = segments[next_layer_segs] 
        lid1, lid2 = lid1[next_layer_segs], lid2[next_layer_segs]
        #print("segs after nextlayer", segments)
        #print(list(zip(lid1, lid2)))
        
        # extract geometric quantities
        r_1 = hits['r'].to_numpy()[segments[:,0]]
        r_2 = hits['r'].to_numpy()[segments[:,1]]
        phi_1 = hits['phi'].to_numpy()[segments[:,0]]
        phi_2 = hits['phi'].to_numpy()[segments[:,1]]
        z_1 = hits['z'].to_numpy()[segments[:,0]]
        z_2 = hits['z'].to_numpy()[segments[:,1]]
        eta_1, eta_2 = calc_eta(r_1, z_1), calc_eta(r_2, z_2)
        dR = np.sqrt((phi_2-phi_1)**2 + (eta_2-eta_1)**2)
        
        
        # apply geometric edge cuts
        dr = r_2 - r_1
        dphi = calc_dphi(phi_1, phi_2)
        dz = z_2 - z_1
        z0 = z_1 - r_1 * dz/dr
        phi_slope = dphi/dr
        good_seg_mask = (abs(phi_slope) < 0.0006) & (abs(z0) < 1500)

        l0_to_EC = ((lid1==0) & ((lid2==11) | (lid2==4)))
        l1_to_EC = ((lid1==1) & ((lid2==11) | (lid2==4)))
        intersected_layer = ((l0_to_EC & (abs(71.56298*dz/dr + z0) < 490.975)) | 
                             (l1_to_EC & (abs(115.3781*dz/dr + z0) < 490.975)))
        good_seg_mask = good_seg_mask & (intersected_layer==False)

        segments = segments[good_seg_mask]
        #print("segs after goodseg", segments)
                
        segs_out = np.append(segs_out, segments[:,0]).astype(int)
        segs_in = np.append(segs_in, segments[:,1]).astype(int)
        
        lid1 = hits['layer'].to_numpy().astype(int)[segments[:,0]]
        lid2 = hits['layer'].to_numpy().astype(int)[segments[:,1]]
        pid1 = hits['particle_id'].to_numpy().astype(int)[segments[:,0]]
        pid2 = hits['particle_id'].to_numpy().astype(int)[segments[:,1]]
        lps = list(zip( list(zip(pid1, pid2)), list(zip(lid1, lid2))))
        #print(lps)

        y0 = np.array(pid1==pid2).astype(int) 
        y = np.append(y, y0)
        dr_all.extend(dr)
        dphi_all.extend(dphi)
        dz_all.extend(dz)
        dR_all.extend(dR)
        #print(y)
        #print(segs_in)
        #print(segs_out)

    print(y[y>0.5].shape[0]/n_truth_edges)
    print(y[y>0.5].shape[0]/y.shape[0])
    
    feature_scale = np.array([1000., np.pi, 1000.])
    X = (hits[['r', 'phi', 'z']].values / feature_scale).astype(np.float32)
    n_hits, n_edges = X.shape[0], segs_out.shape[0]
    Ri = np.zeros([n_hits, n_edges], dtype=np.uint8)
    Ro = np.zeros([n_hits, n_edges], dtype=np.uint8)
    Ri[segs_in.astype(np.uint8), np.arange(n_edges)] = 1
    Ro[segs_out.astype(np.uint8), np.arange(n_edges)] = 1
    #Ra = np.zeros([4, n_edges], dtype=np.uint8)
    
    Ra = np.stack((dr_all/feature_scale[0],
                   dphi_all/feature_scale[1],
                   dz_all/feature_scale[2],
                   dR_all))

    filename = os.path.join(outdir, evt_id)
    graph = Graph(X, Ra, Ri, Ro, y)
    save_graph(graph, filename)
    print(filename)

