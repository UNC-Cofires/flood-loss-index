import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import sys
import os

### *** HELPER FUNCTIONS *** ###

def sample_raster(filepath,xy_coords,band=1):
    """
    param: filepath: path to raster
    param: xy_coords: list of tuples specifying xy-coordinates of points to sample. Must be in same crs as raster. 
    param: band: index of raster band to sample (starts at 1)
    returns: values: sampled values of raster at specified coordinates. 
    """
    
    with rio.open(filepath,'r') as src:
        values = np.array([v[0] for v in rio.sample.sample_gen(src, xy_coords,indexes=band)])
        
    return(values)

### *** LOAD DATA SOURCES *** ###

# Specify raster processing unit (RPU) of interest (passed as command-line argument)
RPU=sys.argv[1]
outfolder = f'/proj/characklab/projects/kieranf/flood_damage_index/data/rasters/{RPU}'

# Specify path to raster data files
raster_variables = ['nhd_catchment_comid','elev_cm','dist_wb_m','hand_wb_cm','slope_x1000','geomorphon','tpi_cm','fac']
raster_paths = [os.path.join(outfolder,f'{RPU}_{var}.tif') for var in raster_variables]

# Specify coordinate reference system
crs = 'EPSG:5070'

# Read in building/structure points
structure_info_path = '/proj/characklab/projects/kieranf/flood_damage_index/data/NSI/CONUS_structure_info.parquet'
structure_info = pd.read_parquet(structure_info_path,filters=[('rpu_id','==',RPU)]).sort_values(by=['x_epsg5070','y_epsg5070'])
structures = gpd.GeoDataFrame(structure_info, geometry=gpd.points_from_xy(structure_info['x_epsg5070'], structure_info['y_epsg5070'],crs=crs))

### *** SAMPLE RASTERS *** ###

# Specify coordinates of points to sample
xy_coords = structures.apply(lambda row: (row['x_epsg5070'],row['y_epsg5070']),axis=1).to_list()

# Iterate over rasters
for filepath,variable in zip(raster_paths,raster_variables):
    print(variable,flush=True)
    structures[variable] = sample_raster(filepath,xy_coords)
    
### *** SAVE RESULTS *** ###
structures.drop(columns=['geometry'],inplace=True)
outname = os.path.join(outfolder,f'{RPU}_raster_values_at_structure_points.parquet')
structures.to_parquet(outname)
