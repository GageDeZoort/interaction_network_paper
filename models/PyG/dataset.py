import os
import torch
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, indir, graph_files):
        self.indir = indir
        self.graph_files = graph_files
    
    def __len__(self):
        return len(self.graph_files)
        

    def __getitem__(self, index):
        filename = self.graph_files[index]
        path = os.path.join(self.indir, filename)
        graph = self.load_graph(path)
        x = graph['x']
        edge_attr = np.transpose(graph['edge_attr'])
        data = {'x': x, 'edge_attr': edge_attr, 'edge_index': graph['edge_index']}
        target = {'y': graph['y'], 'pid': graph['pid']}
        return data, target

    def load_graph(self, path):
        with np.load(path) as f:
            x, edge_attr, y, pid = f['X'], f['Ra'], f['y'], f['pid']
            edge_index = np.stack((f['Ri_rows'], f['Ro_rows']))
            return {'x': x, 'edge_attr': edge_attr, 'edge_index': edge_index,
                    'y': y, 'pid': pid}

        
