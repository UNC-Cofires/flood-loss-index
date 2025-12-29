#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=150g
#SBATCH -t 6:00:00
#SBATCH --mail-type=all
#SBATCH --job-name=concat_precip
#SBATCH --mail-user=kieranf@email.unc.edu

module purge
module load anaconda
export PYTHONWARNINGS="ignore"

# Combine event-specific precipitation summaries into one dataframe
conda activate /proj/characklab/projects/kieranf/flood_damage_index/fli-env-v1
python3.12 concatenate_precip.py

