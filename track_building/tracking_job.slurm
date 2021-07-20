#!/bin/bash
#SBATCH --job-name=torch-test-cuda    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=6        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=60G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --time=2:00:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3
conda activate geoenv

python measure_tracking_effs.py 0p9
