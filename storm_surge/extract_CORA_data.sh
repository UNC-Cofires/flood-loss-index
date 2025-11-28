#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=150g
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=all
#SBATCH --job-name=CORA_extract
#SBATCH --mail-user=kieranf@email.unc.edu

module purge
module load anaconda
export PYTHONWARNINGS="ignore"

conda activate /proj/characklab/projects/kieranf/flood_damage_index/fli-env-v1
python3.12 extract_CORA_data.py