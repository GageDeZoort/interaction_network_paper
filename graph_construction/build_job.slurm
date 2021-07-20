#!/bin/bash
#SBATCH --job-name=torch-test    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=20G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=3:00:00           # total run time limit (HH:MM:SS)

module purge
module load anaconda3
conda activate geo_gnnenv

echo "...building geometric"
python build_geometric.py configs/geometric.yaml --start-evtid=1000 --end-evtid=2000
