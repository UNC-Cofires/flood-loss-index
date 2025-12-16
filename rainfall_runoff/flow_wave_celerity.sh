#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64g
#SBATCH -t 6:00:00
#SBATCH --mail-type=all
#SBATCH --job-name=flow_celerity
#SBATCH --mail-user=kieranf@email.unc.edu

module purge
module load anaconda
export PYTHONWARNINGS="ignore"

# Determine grid cell weights for each catchment
conda activate /proj/characklab/projects/kieranf/flood_damage_index/fli-env-v1
python3.12 flow_wave_celerity.py

