import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import rioxarray as rx
from rioxarray.merge import merge_arrays
import xarray as xr
import sys
import os

### *** INITIAL SETUP *** ###

# Specify raster processing unit (RPU) of interest (passed as command-line argument)
RPU = sys.argv[1]
outfolder = f'/proj/characklab/projects/kieranf/flood_damage_index/data/rasters/{RPU}'

# Specify path to RPU shapefile
pwd = os.getcwd()
RPU_path = '/proj/characklab/projects/kieranf/flood_damage_index/data/watersheds/CONUS_RPU_GEC_cleaned'

# Specify path to previously created raster files
study_area_path = os.path.join(outfolder,f'{RPU}_study_area_10km_buffer.tif')

# Specify path to Oak Ridge National Lab (ORNL) 10-meter HAND and elevation rasters
ornl_dir = '/proj/characklab/projects/kieranf/flood_damage_index/data/OakRidgeHAND/HU6_HAND'
ornl_included_hucs = np.sort(os.listdir(ornl_dir))

### *** LOAD DATA SOURCES *** ###

# RPU boundary
study_area = gpd.read_file(RPU_path)
study_area = study_area[study_area['UnitID']==RPU]

# Study area raster and elevation
study_area_rds = rx.open_rasterio(study_area_path,mask_and_scale=True)

# Create study area mask
study_area_mask = (study_area_rds.values==1)

# Get overlapping huc6 basins
included_hucs = study_area['HU6_basins'].values[0].split(',')
included_hucs = [x for x in included_hucs if x in ornl_included_hucs]

### *** ALIGN AND MERGE ORNL RASTERS *** ###

huc_elev_list = []
huc_hand_list = []

# Align arrays
for huc in included_hucs: 
    
    huc_hand_path = os.path.join(ornl_dir,f'{huc}/{huc}hand.tif')
    huc_elev_path = os.path.join(ornl_dir,f'{huc}/{huc}.tif')

    huc_hand_rds = rx.open_rasterio(huc_hand_path,mask_and_scale=True).rio.reproject_match(study_area_rds)
    huc_hand_list.append(huc_hand_rds)
    
    huc_elev_rds = rx.open_rasterio(huc_elev_path,mask_and_scale=True).rio.reproject_match(study_area_rds)
    huc_elev_list.append(huc_elev_rds)

# Merge arrays
ornl_hand_rds = merge_arrays(huc_hand_list)
ornl_elev_rds = merge_arrays(huc_elev_list)

# Mask areas outside of study domain
ornl_hand_rds.values[~study_area_mask] = np.nan
ornl_elev_rds.values[~study_area_mask] = np.nan

### *** SAVE RESULT *** ###

# Specify desired data type and nodata value for rasters
# (Note that we can reduce disk usage by saving things as integers) 
raster_dtype = np.int32
nodata_value = -999999

# Convert elevation from m to cm
ornl_hand_rds = 100*ornl_hand_rds
ornl_elev_rds = 100*ornl_elev_rds

# Replace nan with nodata value
ornl_hand_rds = ornl_hand_rds.fillna(nodata_value)
ornl_elev_rds = ornl_elev_rds.fillna(nodata_value)

# Cast as integer data type
ornl_hand_rds = ornl_hand_rds.astype(raster_dtype)
ornl_elev_rds = ornl_elev_rds.astype(raster_dtype)

# Encode nodata value in the raster metadata
ornl_hand_rds = ornl_hand_rds.rio.write_nodata(nodata_value, encoded=True)
ornl_elev_rds = ornl_elev_rds.rio.write_nodata(nodata_value, encoded=True)

# Save to file 
outname = os.path.join(outfolder,f'{RPU}_ornl_hand_cm.tif')
ornl_hand_rds.rio.to_raster(outname)

outname = os.path.join(outfolder,f'{RPU}_ornl_elev_cm.tif')
ornl_elev_rds.rio.to_raster(outname)