#!/bin/bash
#SBATCH --job-name=torch-test-cuda    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=3        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3
conda activate torch-env

echo "python ../run_track_condensation_network.py -o ../trained_models --outfile=qmin0p01_sb0p25_lr0p0005_g0p95 --n-train=8000 --n-test=2000 --n-val=250 --learning-rate=0.0005 --gamma=0.95 --q-min=0.01 --sb=0.25"
python ../run_track_condensation_network.py -o ../trained_models --outfile=qmin0p01_sb0p25_lr0p0005_g0p95 --n-train=8000 --n-test=2000 --n-val=250 --learning-rate=0.0005 --gamma=0.95 --q-min=0.01 --sb=0.25

