import os
import sys
import numpy as np
import pandas as pd
import trackml
from trackml.dataset import load_event
from trackml.dataset import load_dataset

pixel_layers = [(8,2), (8,4), (8,6), (8,8),
                (7,14), (7,12), (7,10),
                (7,8), (7,6), (7,4), (7,2),
                (9,2), (9,4), (9,6), (9,8),
                (9,10), (9,12), (9,14)]

layer_pairs = [(0,1), (1,2), (2,3), 
               (0,4), (1,4), (2,4),
               (4,5), (5,6), (6,7), (7,8), (8,9), (9,10),
               (0,11), (1,11), (2,11),
               (11,12), (12,13), (13,14), (14,15), (15,16), (16,17)]


pt_min = float(sys.argv[1])
train_sample = 2
indir = '/tigress/jdezoort/train_{}'.format(train_sample)
evtid_base = 'event00000'
evtids = os.listdir(indir) #[evtid_base+str(i) for i in np.arange(1000, , 1)]
evtids = [evtid.split('-')[0] for evtid in evtids if 'hits' in evtid]

module_labels = {}
hits, cells, particles, truth = load_event(os.path.join(indir, evtids[0]))
hits_by_loc = hits.groupby(['volume_id', 'layer_id'])
hits = pd.concat([hits_by_loc.get_group(pixel_layers[i]).assign(layer=i)
                  for i in range(len(pixel_layers))])
for lid, lhits in hits.groupby('layer'):
    module_labels[lid] = np.unique(lhits['module_id'].values)

module_maps = {(i,j): np.zeros((np.max(module_labels[i])+1, np.max(module_labels[j])+1))
               for (i,j) in layer_pairs}
total_connections = []

for i, evtid in enumerate(evtids):
    print(i, evtid)
    hits, cells, particles, truth = load_event(os.path.join(indir, evtid))
    hits_by_loc = hits.groupby(['volume_id', 'layer_id'])
    hits = pd.concat([hits_by_loc.get_group(pixel_layers[i]).assign(layer=i)
                      for i in range(len(pixel_layers))])
    pt = np.sqrt(particles.px**2 + particles.py**2)
    particles['pt'] = pt
    particles = particles[pt > pt_min]

    truth = (truth[['hit_id', 'particle_id']]
             .merge(particles[['particle_id', 'pt']], on='particle_id'))

    r = np.sqrt(hits.x**2 + hits.y**2)
    phi = np.arctan2(hits.y, hits.x)

    hits = (hits[['hit_id', 'z', 'layer', 'module_id']]
            .assign(r=r, phi=phi)
            .merge(truth[['hit_id', 'particle_id', 'pt']], on='hit_id'))
    
    hits = (hits.loc[
            hits.groupby(['particle_id', 'layer'], as_index=False).r.idxmin()
        ]).assign(evtid=evtid)
    
    hits_by_loc = hits.groupby('layer')
    for lp in layer_pairs:
        hits0 = hits_by_loc.get_group(lp[0])
        hits1 = hits_by_loc.get_group(lp[1])
        keys = ['evtid', 'particle_id', 'module_id', 'r', 'phi', 'z']
        hit_pairs = hits0[keys].reset_index().merge(
            hits1[keys].reset_index(), on='evtid', suffixes=('_1', '_2'))
        pid1, pid2 = hit_pairs['particle_id_1'], hit_pairs['particle_id_2']
        hit_pairs = hit_pairs[pid1==pid2]
        mid1, mid2 = hit_pairs['module_id_1'].values, hit_pairs['module_id_2'].values
        r1, r2 = hit_pairs['r_1'].values, hit_pairs['r_2'].values
        for i in range(len(mid1)):
            module_maps[lp][mid1[i]][mid2[i]]+=1
    
    connections = 0
    for module_map in module_maps.values():
        connections += np.sum(module_map > 0)
    total_connections.append(connections)


pt_lookup = {0.5: '0p5', 0.6: '0p6', 0.7: '0p7', 0.8: '0p8',
             0.9: '0p9', 1.0: '1', 1.1: '1p1', 1.2: '1p2',
             1.3: '1p3', 1.4: '1p4', 1.5: '1p5', 1.6: '1p6', 
             1.7: '1p7', 1.8: '1p8', 1.9: '1p9', 2.0: '2'}

pt_str = pt_lookup[pt_min]
with open(f'module_map_{train_sample}_{pt_str}GeV.npy', 'wb') as f:
    np.save(f, module_maps)
