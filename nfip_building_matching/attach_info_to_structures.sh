#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=150g
#SBATCH -t 5-00:00:00
#SBATCH --mail-type=all
#SBATCH --job-name=attach_info_to_structures
#SBATCH --mail-user=kieranf@email.unc.edu
#SBATCH --array=0-48%15

module purge
module load anaconda
export PYTHONWARNINGS="ignore"

conda activate /proj/characklab/projects/kieranf/flood_damage_index/fli-env-v1
python3.12 attach_info_to_structures.py