#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00          # Maximum run time in hh:mm:ss
#SBATCH --mem=54gb             # Maximum memory required (in megabytes)
#SBATCH --job-name=default_479  # Job name (to track progress)
#SBATCH --partition=cse479      # Partition on which to run job
#SBATCH --gres=gpu:1            # Requests a GPU

#SBATCH --error=model_training_%J.err
#SBATCH --output=model_training_%J.out
module purge

module load anaconda
conda activate tensorflow-env

python main.py
# This line runs everything that is put after "sbatch submit_gpu.sh ..."
# $@