#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=36g
#SBATCH -t 3-00:00:00
#SBATCH --mail-type=all
#SBATCH --job-name=summarize_precip
#SBATCH --mail-user=kieranf@email.unc.edu
#SBATCH --array=1-41%20

module purge
module load anaconda
export PYTHONWARNINGS="ignore"

# Summarize precipitation for each event
conda activate /proj/characklab/projects/kieranf/flood_damage_index/fli-env-v1
python3.12 summarize_upstream_precip_by_event.py
