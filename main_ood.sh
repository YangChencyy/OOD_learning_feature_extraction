#!/bin/bash

#SBATCH --account=kusari0
#SBATCH --job-name=OOD_GP
#SBATCH --mail-user=rivachen@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=10GB
#SBATCH --time=13:00:00

# module purge
# conda init bash
# conda activate GP

python main.py > output.txt