#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=64g
#SBATCH -t 5-00:00:00
#SBATCH --mail-type=all
#SBATCH --job-name=annual_maxima
#SBATCH --mail-user=kieranf@email.unc.edu
#SBATCH --array=1-9

module purge
module load anaconda
export PYTHONWARNINGS="ignore"

conda activate /proj/characklab/projects/kieranf/flood_damage_index/fli-env-v1

# Annual max 24-hour precipitation intensity
python3.12 extract_annual_maxima.py 24
