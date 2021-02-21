"""
This module contains code for interacting with hit graphs.
A Graph is a namedtuple of matrices X, Ra, Ri, Ro, y, pid.
"""

from time import time
from collections import namedtuple
import numpy as np

Graph = namedtuple('Graph', ['X', 'Ra', 'Ri', 'Ro', 'y', 'pid'])

def graph_to_sparse(graph):
    Ri_rows, Ri_cols = graph.Ri.nonzero()
    Ro_rows, Ro_cols = graph.Ro.nonzero()
    return dict(X=graph.X, Ra=graph.Ra, 
                Ri_rows=Ri_rows, Ri_cols=Ri_cols,
                Ro_rows=Ro_rows, Ro_cols=Ro_cols,
                y=graph.y, pid=graph.pid)

def sparse_to_graph(X, Ra, Ri_rows, Ri_cols, Ro_rows, Ro_cols, y, pid, dtype=np.uint8):
    n_nodes, n_edges = X.shape[0], Ri_rows.shape[0]
    Ri = np.zeros((n_nodes, n_edges), dtype=dtype)
    Ro = np.zeros((n_nodes, n_edges), dtype=dtype)
    Ri[Ri_rows, Ri_cols] = 1
    Ro[Ro_rows, Ro_cols] = 1
    return Graph(X, Ra, Ri, Ro, y, pid)

def save_graph(graph, filename):
    """Write a single graph to an NPZ file archive"""
    np.savez(filename, **graph_to_sparse(graph))

def save_graphs(graphs, filenames):
    for graph, filename in zip(graphs, filenames):
        save_graph(graph, filename)

def load_graph(filename):
    """Read a single graph NPZ"""
    with np.load(filename) as f:
        return sparse_to_graph(**dict(f.items()))
    #with np.load(filename) as f:
    #    return Graph(**dict(f.items()))

def load_graphs(filenames, graph_type=Graph):
    return [load_graph(f) for f in filenames]
