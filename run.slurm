#!/bin/bash
# --------------------------- cheatsheet.slurm ---------------------------
#SBATCH --job-name=flowcat
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=g101eth0
#SBATCH --time=10:00:00
#SBATCH --gres gpu:1
#SBATCH --error=logs/job.%J.out
#SBATCH --output=logs/job.%J.out

module load anaconda/4.6.11-py37
# module load anaconda/4.7.12-py27
conda activate flowcat

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Current working directory is `pwd`"

script="$(realpath $1)"
# 
source "$script"
echo "Finished running $script."
# --------------------------- cheatsheet.slurm ---------------------------
