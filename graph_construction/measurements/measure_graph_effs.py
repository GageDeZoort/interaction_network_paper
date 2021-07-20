#!/usr/bin/env

import os
import sys
import math
import argparse
import pickle
from os import listdir
from os.path import isfile, join

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

import trackml
from trackml.dataset import load_event
from trackml.dataset import load_dataset

# break down input files into event_ids 
data_dir = '/tigress/jdezoort/train_1'
files = [f for f in listdir(data_dir)]
files = [f.split('.')[0] for f in files if "hits" in f]
evt_ids = [f.split('-')[0] for f in files]

N_avg = 1770
endcaps = True

parser = argparse.ArgumentParser(description='Measure Graph Construction Efficiencies')
parser.add_argument('--pt-min', type=float, default=2, 
                    help='pt_min for graph construction')
parser.add_argument('--construction', type=str, default='geometric', 
                    help='construction method: geometric, pre-clustering, or data-driven')
parser.add_argument('--train-sample', type=int, default=1,
                    help='train_<train-sample> used to generate graphs')
parser.add_argument('--N', type=int, default=1770,
                    help='number of graphs use in calculation')
args = parser.parse_args()
print(' ... using args:\n', args)

pt_cut = args.pt_min
method = args.construction

pixel_layers = [(8,2), (8,4), (8,6), (8,8)]
valid_layer_pairs = [(0,1), (1,2), (2,3)]
pixel_layers.extend([(7,2), (7,4), (7,6), (7,8),
                     (7,10), (7,12), (7,14),
                     (9,2), (9,4), (9,6), (9,8),
                     (9,10), (9,12), (9,14)])
valid_layer_pairs.extend([(0,4), (1,4), (2,4), (3,4),
                          (0,11), (1,11), (2,11), (3,11),
                          (4,5), (5,6), (6,7),
                          (7,8), (8,9), (9,10),
                          (11,12), (12,13), (13,14),
                          (14,15), (15,16), (16,17)])

pt_cut_str = {0.5: '0p5', 0.6: '0p6', 0.7: '0p7', 0.8: '0p8',  0.9: '0p9', 1: '1', 
              1.1: '1p1', 1.2: '1p2', 1.3: '1p3', 1.4: '1p4', 1.5: '1p5', 1.6: '1p6',
              1.7: '1p7', 1.8: '1p8', 1.9: '1p9', 2: '2'}
graph_dir = '../../../hitgraphs_{}/{}_{}/'.format(args.train_sample, method, 
                                                  pt_cut_str[pt_cut])
print(" ... reading graphs from", graph_dir, '\n')

truth = {}
purities, efficiencies = [], []
sizes, nodes, edges = [], [], []
counter = 0
for i, evtid in enumerate(evt_ids):
    #if (int(evtid.split("00000")[1].split(".")[0]) > 2820+N_avg): continue

    # load in graph 
    graph_path = graph_dir + evtid + '_g000.npz'
    if not os.path.isfile(graph_path):
        continue
        
    print(evtid)
    with np.load(graph_path) as f:
        x = torch.from_numpy(f['x'])
        edge_attr = torch.from_numpy(f['edge_attr'])
        edge_index = torch.from_numpy(f['edge_index'])
        y = torch.from_numpy(f['y'])
        pid = torch.from_numpy(f['pid'])
        data = Data(x=x, edge_index=edge_index,
                    edge_attr=torch.transpose(edge_attr, 0, 1),
                    y=y, pid=pid)
        data.num_nodes = len(x)
        
    size = sys.getsizeof(data.x) + sys.getsizeof(data.edge_attr) 
    size += sys.getsizeof(data.edge_index) + sys.getsizeof(data.y)
        
    #print("x: {}, edge_attr: {}, edge_index: {}, y: {}"
    #      .format(data.x.shape, data.edge_attr.shape, data.edge_index.shape, data.y.shape))

    n_edges, n_nodes = edge_index.shape[1], x.shape[0]

    #print("n_edges: {}, n_nodes: {}".format(n_edges, n_nodes))

    hits, cells, particles, truth = load_event(os.path.join(data_dir, evtid))
    hits_by_loc = hits.groupby(['volume_id', 'layer_id'])
    hits = pd.concat([hits_by_loc.get_group(pixel_layers[i]).assign(layer=i)
                      for i in range(len(pixel_layers))])
    
    n_particles_before = particles.particle_id.unique().shape[0]
    pt = np.sqrt(particles.px**2 + particles.py**2)
    particles = particles[pt > pt_cut]
    
    truth = (truth[['hit_id', 'particle_id', 'weight']]
             .merge(particles[['particle_id']], on='particle_id'))
    
    r = np.sqrt(hits.x**2 + hits.y**2)
    phi = np.arctan2(hits.y, hits.x)
    
    hits = (hits[['hit_id', 'z', 'layer']]
            .assign(r=r, phi=phi, evtid=evtid)
            .merge(truth[['hit_id', 'particle_id', 'weight']], on='hit_id'))
    hits = hits.loc[
        hits.groupby(['particle_id', 'layer'], as_index=False).r.idxmin()
    ]
    
    particle_ids = hits.particle_id.unique()
    n_particles_after = particle_ids.shape[0]

    particle_list = hits.groupby('particle_id')
    
    truth_edges_per_particle = {}
    truth_edges =  0
    for p_id, particle in particle_list:
        if (p_id == 0): continue
        hit_ids = np.unique(particle.hit_id)
        n_layers_hit = len(hit_ids)
        
        if (n_layers_hit > 1): 
            layers = np.array(particle.layer.values)
            lo, li = layers[:-1], layers[1:]
            layer_pairs = np.column_stack((lo, li))
          
            for lp in layer_pairs:
                if (valid_layer_pairs==lp).any():
                    #truth_edges += n_layers_hit-1
                    truth_edges += 1
                else:
                    print(lp, 'not in valid pairs!')
        
        truth_edges_per_particle[p_id] = n_layers_hit-1

    efficiency = torch.sum(data.y).item()/truth_edges
    purity = torch.sum(data.y).item()/len(data.y)
    print(' > efficiency: {:.4f}/{:.4f}={:.4f}'.format(torch.sum(y).item(), truth_edges, efficiency))
    print(' > purity: {:.4f}/{:.4f}={:.4f}'.format(torch.sum(y).item(), len(y), purity))
    print(' > n_particles={} (pt > {} GeV) / {}'
          .format(n_particles_after, pt_cut, n_particles_before))
    print(' > graph edges={}\n > graph nodes={}'
          .format(n_edges, n_nodes))
    
    #print(' > size={}\n'.format(size/10**6))
    if (torch.sum(y).item()/truth_edges > 1.0): print('\nERROR: PURITY>1!\n')

    purities.append(purity)
    efficiencies.append(efficiency)
    sizes.append(size)
    nodes.append(n_nodes)
    edges.append(n_edges)
    counter += 1
    if counter > int(args.N): continue

print(' ********************* ')
print(' ** AVERAGE RESULTS ** ')
print(' ********************* ')
print(' > efficiency = {:.4f}+/-{:.4f}'
      .format(np.mean(efficiencies), np.std(efficiencies)))
print(' > purity = {:.4f}+/-{:.4f}'
      .format(np.mean(purities), np.std(purities)))
print(' > nodes={:.3f}+/-{:.3f}\n > edges={:.3f}+/-{:.3f}'
      .format(np.mean(nodes), np.std(nodes),
              np.mean(edges), np.std(edges)))
#sizes = np.array(sizes)/10**6
#print('avg size={:.3f}+/-{:.3f}'
#      .format(sizes.mean(), sizes.std()))
