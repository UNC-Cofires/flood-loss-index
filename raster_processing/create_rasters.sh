#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=150g
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=all
#SBATCH --job-name=create_rasters
#SBATCH --mail-user=kieranf@email.unc.edu
#SBATCH --array=1-59

# Determine raster processing unit (RPU) of interest based on task_id
RPU="$(sed -n ${SLURM_ARRAY_TASK_ID}p /proj/characklab/projects/kieranf/flood_damage_index/analysis/CONUS_RPU_list.txt)"
echo $RPU

module purge
module load anaconda
export PYTHONWARNINGS="ignore"

#Initial processing of NHD rasters
conda activate /proj/characklab/projects/kieranf/flood_damage_index/fli-env-v1
python3.12 initial_processing.py $RPU
conda deactivate

#Run GRASS GIS script to calculate slope and terrain forms from DEM
apptainer exec --bind /proj/characklab/projects/kieranf/flood_damage_index/ /proj/characklab/projects/kieranf/GRASS/grass-gis_releasebranch_8_4-debian.sif grass --tmp-project EPSG:5070 --exec bash $PWD/grass_script.sh $RPU

#Calculate HAND and stream distance using custom functions
conda activate /proj/characklab/projects/kieranf/flood_damage_index/f2py_env
python3.11 flowpath_calculations.py $RPU
conda deactivate

# Sample rasters at structure points
conda activate /proj/characklab/projects/kieranf/flood_damage_index/fli-env-v1
python3.12 sample_raster_data.py $RPU
conda deactivate








