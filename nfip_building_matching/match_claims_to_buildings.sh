#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32g
#SBATCH -t 3-00:00:00
#SBATCH --mail-type=all
#SBATCH --job-name=match_claims
#SBATCH --mail-user=kieranf@email.unc.edu
#SBATCH --array=37,25,41

module purge
module load anaconda
export PYTHONWARNINGS="ignore"

conda activate /proj/characklab/projects/kieranf/flood_damage_index/fli-env-v1
python3.12 match_claims_to_buildings.py