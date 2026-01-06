#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=300g
#SBATCH -t 3-00:00:00
#SBATCH --mail-type=all
#SBATCH --job-name=create_weightmaps
#SBATCH --mail-user=kieranf@email.unc.edu

module purge
module load anaconda
export PYTHONWARNINGS="ignore"

conda activate /proj/characklab/projects/kieranf/flood_damage_index/fli-env-v1

# Average 24-hour precipitation intensity with return period of 2 years
python3.12 extreme_precip_analysis.py 24 2