# Input/output configuration
input_dir: /tigress/jdezoort/train_1
output_dir:  /scratch/gpfs/jdezoort/hitgraphs/heptrkx_classic_0p75
n_files: 1770

# Graph building configuration
selection:
    pt_min: 0.75 # GeV
    phi_slope_max: 0.0006
    z0_max: 100
    n_phi_sections: 1
    n_eta_sections: 1
    eta_range: [-5, 5]
