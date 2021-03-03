import os
from time import time
import torch
import numpy as np
from models.graph import Graph, save_graphs, load_graph

class Dataset(torch.utils.data.Dataset):
    def __init__(self, indir, graph_files):
        self.indir = indir
        self.graph_files = graph_files
    
    def __len__(self):
        return len(self.graph_files)
        

    def __getitem__(self, index):
        filename = self.graph_files[index]
        path = os.path.join(self.indir, filename)
        t0 = time()
        graph = load_graph(path)
        X = np.transpose(graph.X)
        Ra = np.transpose(graph.Ra)
        #Ra = np.zeros((len(graph.y), 1))
        Ri = np.transpose(graph.Ri)
        Ro = np.transpose(graph.Ro)
        return {'X': X, 'Ra': Ra, 'Ri': Ri, 'Ro': Ro}, {'y': graph.y, 'pid': graph.pid}
