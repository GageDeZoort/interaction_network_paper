# Charged Particle Tracking via Edge-Classifying Interacting Networks
## Repo Organization
- **graph_construction/**
  - build_heptrkx_classic.py: HEP.TrkX construction algorithm to build graphs in the pixel barrel layers, example usage: 
    ``` 
    python build_heptrkx_classic.py configs/heptrkx_classic.py
    ```
  - build_heptrkx_plus.py: HEP.TrkX construction algorithm modified to build graphs in the pixel barrel+endcaps layers, example usage: 
    ```
    python build_heptrkx_plus.py configs/heptrkx_plus.py
    ```
  - build_dbgraphs.py: DBScan clustering algorithm used build graphs via clustering in eta-phi space, example usage: 
    ```
    python build_dbgraphs.py configs/dbgraphs.py
     ```
- **models/** 
  - interaction_network.py: network architecture 
  - dataset.py: custom data loader for multi-threaded graph loading (necessary for large graphs)
  - track_fitter.py: class to fit hits in conformal space 
  
- run_interaction_network.py: IN training script, example usage:
  ```
  python run_interaction_network.py --pt=1 --construction=heptrkx_plus --lr=0.005 --gamma=0.9 --save-model 
  ```

- timing_scan_bash.py: inference timing script, example usage:
  - CPU-only inference:
    ```
    python timing_scan_bash.py --construction=heptrkx_plus_pyg --graphs=5 --batchsize=1 --loops=100 --repeat=5
    ```
  - GPU inference:
    ```
    python timing_scan_bash.py --gpu --construction=heptrkx_plus_pyg --graphs=5 --batchsize=1 --loops=100 --repeat=5
    ```
  - Note, the code assumes some hitgraphs are located in the following directory structure
```
hitgraphs
├── heptrkx_plus_pyg_0GeV5
├── heptrkx_plus_pyg_0GeV6
├── heptrkx_plus_pyg_0GeV75
├── heptrkx_plus_pyg_1GeV
└── heptrkx_plus_pyg_1GeV
```