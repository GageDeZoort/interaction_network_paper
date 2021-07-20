# Charged Particle Tracking via Edge-Classifying Interacting Networks
## Repo Organization
- **models/** 
Several classes are defined to implement the interaction network and facilitate graph input/output:
  - interaction_network.py: network architecture implemented in PyTorch Geometric (PyG)
  - dataset.py: custom data loader for multi-threaded graph loading in the PyG Data representation
  
- **Graph Construction**:
Graph construction algorithms (in ``graph_construction/``) process TrackML samples (e.g. `train_1`) into hitgraphs, which are defined as tuples of node features, edge features, truth labels, and edge indices in COOrdinate format. Several graph construction algorithms are available, each of which is configured by a set of particle-specific and geometric edge selections: 
  - build_geometric.py: geometric edge selection algorithm to build graphs in the pixel barrel layers, example usage: 
    ```bash 
    python build_geometric.py configs/geometric.py
    ```
  - build_pre-clustering.py: DBSCAN clustering algorithm used build graphs via clustering in eta-phi space, example usage: 
    ```bash
    python build_pre-clustering.py configs/pre-clustering.py
     ```
  - build_data-driven.py: module map method used to build graphs based on previously observed data, example usage:
    ```bash
    python build_pre-clustering.py configs/pre-clustering.py
     ```
     - Note that a module map is necessary to run this graph construction method. Module maps are built and stored in the `graph_construction/module_maps` directory. Note that an orthogonal data sample should be used to construct the module map; for example, to build graphs from `train_1`, a module map from `train_<2-5>`. 

- **Training and Inference**:
  - `run_interaction_network.py`: IN training script, example usage:
  ```bash
  python run_interaction_network.py --pt=1 --construction=heptrkx_plus --lr=0.005 --gamma=0.9 --save-model 
  ```
  - `gnn_inference/` contains a script for testing the classification accuracy of a set of trained models 
  - `timing_scan_bash.py`: inference timing script, example usage:
    - CPU-only inference:
    ```bash
    python timing_scan_bash.py --construction=heptrkx_plus_pyg --graphs=5 --batchsize=1 --loops=100 --repeat=5
    ```
    - GPU inference:
    ```bash
    python timing_scan_bash.py --gpu --construction=heptrkx_plus_pyg --graphs=5 --batchsize=1 --loops=100 --repeat=5
    ```
  - Note, the code assumes some hitgraphs are located in the following directory structure
    ```
    hitgraphs
    ├── heptrkx_plus_pyg_0GeV5
    ├── heptrkx_plus_pyg_0GeV6
    ├── heptrkx_plus_pyg_0GeV7
    ├── heptrkx_plus_pyg_0GeV8
    ├── heptrkx_plus_pyg_0GeV9
    ├── heptrkx_plus_pyg_1GeV
    ├── heptrkx_plus_pyg_1GeV5
    └── heptrkx_plus_pyg_2GeV
    ```
 - **Track Building**: 
    - Track building algorithms cluster hits in the edge-weigted graphs. Several tracking efficiency definitions are implemented in ``track_building/measure_tracking_effs.py`` and are reported per sample, per pt, and per eta. 
