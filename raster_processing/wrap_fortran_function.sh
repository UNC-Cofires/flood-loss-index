#!/bin/bash
module purge
module load anaconda
conda activate /proj/characklab/projects/kieranf/flood_damage_index/f2py_env
python3.11 -m numpy.f2py -c -m diffusion diffusion.f90
