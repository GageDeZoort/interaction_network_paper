#!/bin/bash
#SBATCH --job-name=torch-test-cuda    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=6        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=10:00:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3
conda activate torch-env

echo "2/2: R1=30, R2=30, O=30, ReLU; python run_interaction_network.py --epochs=120 --lr=0.005 --gamma=0.9 --pt=1 --construction=heptrkx_plus --save-model"
python run_interaction_network.py --epochs=200 --lr=0.005 --gamma=0.9 --pt=1 --step-size=10 --construction=heptrkx_plus_pyg
