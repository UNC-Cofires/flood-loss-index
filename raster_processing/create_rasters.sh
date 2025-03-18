#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16g
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=all
#SBATCH --job-name=create_rasters
#SBATCH --mail-user=kieranf@email.unc.edu
#SBATCH --array=177

HUC6="$(sed -n ${SLURM_ARRAY_TASK_ID}p /proj/characklab/projects/kieranf/flood_damage_index/analysis/CONUS_HUC6_list.txt)"
echo $HUC6

module purge
module load anaconda
export PYTHONWARNINGS="ignore"
conda activate /proj/characklab/projects/kieranf/flood_damage_index/fli-env-v1

python3.12 initial_processing.py $HUC6




