#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=90g
#SBATCH -t 0-03:00:00
#SBATCH --mail-type=all
#SBATCH --job-name=max_zeta_by_event
#SBATCH --mail-user=kieranf@email.unc.edu

module purge
module load anaconda
export PYTHONWARNINGS="ignore"

conda activate /proj/characklab/projects/kieranf/flood_damage_index/fli-env-v1
python3.12 get_peak_over_threshold.py
