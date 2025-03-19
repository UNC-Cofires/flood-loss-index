#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=90g
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=all
#SBATCH --job-name=create_rasters
#SBATCH --mail-user=kieranf@email.unc.edu
#SBATCH --array=177

# Determine HUC6 watershed of interest based on task_id
HUC6="$(sed -n ${SLURM_ARRAY_TASK_ID}p /proj/characklab/projects/kieranf/flood_damage_index/analysis/CONUS_HUC6_list.txt)"
echo $HUC6

module purge
module load anaconda
export PYTHONWARNINGS="ignore"

# Initial processing
conda activate /proj/characklab/projects/kieranf/flood_damage_index/fli-env-v1
python3.12 initial_processing.py $HUC6

# Run GRASS GIS script to calculate distance to and height above waterbodies and floodplains
conda deactivate
apptainer exec --bind /proj/characklab/projects/kieranf/flood_damage_index/ /proj/characklab/projects/kieranf/GRASS/grass-gis_releasebranch_8_4-debian.sif grass --tmp-project EPSG:6350 --exec bash $PWD/grass_script.sh $HUC6

# Diffusion interpolation of floodplain elevations 
conda activate /proj/characklab/projects/kieranf/flood_damage_index/f2py_env
python3.11 diffusion_interpolation.py $HUC6










