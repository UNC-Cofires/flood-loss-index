#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=150g
#SBATCH -t 3-00:00:00
#SBATCH --mail-type=all
#SBATCH --job-name=extreme_precip
#SBATCH --mail-user=kieranf@email.unc.edu
#SBATCH --array=4-9

# Determine raster processing unit (RPU) of interest based on task_id
RPU="$(sed -n ${SLURM_ARRAY_TASK_ID}p /proj/characklab/projects/kieranf/flood_damage_index/analysis/CONUS_RPU_list.txt)"
echo "Raster processing unit: ${RPU}"

module purge
module load anaconda
export PYTHONWARNINGS="ignore"

conda activate /proj/characklab/projects/kieranf/flood_damage_index/fli-env-v1

# Average 24-hour precipitation intensity with return period of 2 years
python3.12 extreme_precip_analysis.py $RPU 24 2
