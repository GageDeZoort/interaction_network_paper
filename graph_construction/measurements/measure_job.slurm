#!/bin/bash
#SBATCH --job-name=0p5           # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3
conda activate geoenv

echo "train_1, 1 GeV, 5 graphs, pre-clustering"
python measure_graph_effs.py --pt-min 1 --construction pre-clustering --train-sample 1 --N 5

