#!/usr/bin/env

import os
import sys
import math
import argparse
import pickle
from os import listdir
from os.path import isfile, join
sys.path.append("../")
sys.path.append("../../")

import numpy as np
import pandas as pd

import trackml
from trackml.dataset import load_event
from trackml.dataset import load_dataset
from models.graph import Graph, load_graph


# break down input files into event_ids 
data_dir = '/tigress/jdezoort/train_2'
files = [f for f in listdir(data_dir)]
files = [f.split('.')[0] for f in files if "hits" in f]
evt_ids = [f.split('-')[0] for f in files]

N_avg = 1770
endcaps = True
pt_cut = float(sys.argv[1])

method = 'heptrkx_plus'
endcaps = False if (method == 'heptrkx_classic') else True

pixel_layers = [(8,2), (8,4), (8,6), (8,8)]
valid_layer_pairs = [(0,1), (1,2), (2,3)]
if (endcaps):
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


pt_cut_str = {0.5: '0p5', 0.6: '0p6', 0.75: '0p75', 1: '1', 1.5: '1p5', 2: '2'}
graph_dir = '/scratch/gpfs/jdezoort/hitgraphs_2/{}_{}/'.format(method, pt_cut_str[pt_cut])

truth = {}
purities, efficiencies = [], []
sizes, nodes, edges = [], [], []
for i, evtid in enumerate(evt_ids):
    #if (int(evtid.split("00000")[1].split(".")[0]) > 1010): continue

    print("evtid", evtid)
    #if (i == N_avg): break
    print('...', evtid)

    # load in graph 
    graph_path = graph_dir + evtid + '_g000.npz'
    graph = load_graph(graph_path)
    X, Ra = graph.X, graph.Ra
    Ri, Ro = graph.Ri, graph.Ro
    y = graph.y
    size = sys.getsizeof(X) + sys.getsizeof(Ra) 
    size += sys.getsizeof(Ri) + sys.getsizeof(Ro) + sys.getsizeof(y)

    for j, Ri_row in enumerate(Ri):
        Ri_row = Ri_row[y > 0.5]
        if(np.sum(Ri_row) > 1): 
            print("ERREREREREREROR!!!!")
        
    print("graph.X: {}, graph.Ra: {}, graph.Ri: {}, graph.y: {}"
          .format(graph.X.shape, graph.Ra.shape, graph.Ri.shape, graph.y.shape))

    n_edges, n_nodes = Ri.shape[0], Ri.shape[1]

    print("n_edges: {}, n_nodes: {}".format(n_edges, n_nodes))

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

        #print(particle)
        #print(hit_ids)
        #print(n_layers_hit-1)
        

    purity = np.sum(y)/truth_edges
    efficiency = np.sum(y)/len(y)
    print('purity: {}/{}={}'.format(np.sum(y), truth_edges, purity))
    print('efficiency: {}/{}={}'.format(np.sum(y), len(y), efficiency))
    print('n_parts={}/{}'
          .format(n_particles_after, n_particles_before))
    print('graph edges={}, graph nodes={}'
          .format(n_edges, n_nodes))
    print('size={}\n'.format(size/10**6))
    if (np.sum(y)/truth_edges > 1.0): print('\nERROR: PURITY>1!\n')

    purities.append(purity)
    efficiencies.append(efficiency)
    sizes.append(size)
    nodes.append(n_nodes)
    edges.append(n_edges)

print('\navg. purity = {:.3f}+/-{:.3f}'
      .format(np.mean(purities), np.std(purities)))
print('avg efficiency = {:.3f}+/-{:.3f}\n'
      .format(np.mean(efficiencies), np.std(efficiencies)))
print('avg nodes={:.3f}+/-{:.3f}, edges={:.3f}+/-{:.3f}'
      .format(np.mean(nodes), np.std(nodes),
              np.mean(edges), np.std(edges)))
sizes = np.array(sizes)/10**6
print('avg size={:.3f}+/-{:.3f}'
      .format(sizes.mean(), sizes.std()))
