import os
import torch
import numpy as np
from torch_geometric.data import Data, Dataset

class GraphDataset(Dataset):
    def __init__(self, transform=None, pre_transform=None,
                 graph_files=[]):
        super(GraphDataset, self).__init__(None, transform, pre_transform)
        self.graph_files = graph_files
    
    @property
    def raw_file_names(self):
        return self.graph_files

    @property
    def processed_file_names(self):
        return []

    def len(self):
        return len(self.graph_files)
        
    def get(self, idx):
        with np.load(self.graph_files[idx]) as f:
            #x, edge_attr, y, pid = f['X'], f['Ra'], f['y'], f['pid']
            x = torch.from_numpy(f['x'])#, dtype=torch.float32)
            edge_attr = torch.from_numpy(f['edge_attr'])#, dtype=torch.float32)
            #Ro_rows = torch.from_numpy(f['Ro_rows'])#, dtype=torch.uint8)
            #Ri_rows = torch.from_numpy(f['Ri_rows'])#, dtype=torch.uint8)
            #edge_index = torch.stack((Ro_rows, Ri_rows))
            edge_index = torch.from_numpy(f['edge_index'])
            y = torch.from_numpy(f['y'])#, dtype=torch.uint8)
            pid = torch.from_numpy(f['pid'])#, dtype=torch.uint8)

            data = Data(x=x, edge_index=edge_index,
                        edge_attr=torch.transpose(edge_attr, 0, 1),
                        y=y, pid=pid)
            data.num_nodes = len(x)

        return data        
