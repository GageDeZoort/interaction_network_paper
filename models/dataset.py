import os
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import ToUndirected

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
            x = torch.from_numpy(f['x'])
            edge_attr = torch.from_numpy(f['edge_attr'])
            edge_index = torch.from_numpy(f['edge_index'])
            y = torch.from_numpy(f['y'])
            pid = torch.from_numpy(f['pid'])
            pt = torch.from_numpy(f['pt']) if 'pt' in f else 0
            eta = torch.from_numpy(f['eta']) if 'eta' in f else 0

            # make graph undirected
            row, col = edge_index
            row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
            edge_index = torch.stack([row, col], dim=0)
            edge_attr = torch.cat([edge_attr, -1*edge_attr], dim=1)
            y = torch.cat([y,y])

            data = Data(x=x, edge_index=edge_index,
                        edge_attr=torch.transpose(edge_attr, 0, 1),
                        y=y, pid=pid, pt=pt, eta=eta)
            data.num_nodes = len(x)

        return data        
