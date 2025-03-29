#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=90g
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=all
#SBATCH --job-name=create_rasters
#SBATCH --mail-user=kieranf@email.unc.edu
#SBATCH --array=10

# Determine raster processing unit (RPU) of interest based on task_id
RPU="$(sed -n ${SLURM_ARRAY_TASK_ID}p /proj/characklab/projects/kieranf/flood_damage_index/analysis/CONUS_RPU_list.txt)"
echo $RPU

module purge
module load anaconda
export PYTHONWARNINGS="ignore"

# Initial processing of NHD rasters
conda activate /proj/characklab/projects/kieranf/flood_damage_index/fli-env-v1
python3.12 initial_processing.py $RPU

# Run GRASS GIS script to calculate distance to and height above waterbodies
conda deactivate
apptainer exec --bind /proj/characklab/projects/kieranf/flood_damage_index/ /proj/characklab/projects/kieranf/GRASS/grass-gis_releasebranch_8_4-debian.sif grass --tmp-project EPSG:5070 --exec bash $PWD/grass_script.sh $RPU









