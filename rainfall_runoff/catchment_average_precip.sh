#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64g
#SBATCH -t 11-00:00:00
#SBATCH --mail-type=all
#SBATCH --job-name=avg_prcp
#SBATCH --mail-user=kieranf@email.unc.edu
#SBATCH --array=4-9

# Determine raster processing unit (RPU) of interest based on task_id
RPU="$(sed -n ${SLURM_ARRAY_TASK_ID}p /proj/characklab/projects/kieranf/flood_damage_index/analysis/CONUS_RPU_list.txt)"
echo $RPU

module purge
module load anaconda
export PYTHONWARNINGS="ignore"

# Average gridded precipitation timeseries over catchments
conda activate /proj/characklab/projects/kieranf/flood_damage_index/fli-env-v1
python3.12 catchment_average_precip.py $RPU 1979 2024


