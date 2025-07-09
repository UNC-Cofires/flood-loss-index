import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from itertools import product
from shapely.geometry import box
import sys
import os

### *** HELPER FUNCTIONS *** ###

def create_regular_grid(x_points,y_points,crs='EPSG:4326'):
    """
    param: x_points: numpy array of grid cell x coordinates (e.g., longitude). Assumes uniform spacing. 
    param: y_points: numpy array of grid cell y coordinates (e.g., latitude). Assumes uniform spacing. 
    param: crs: coordinate reference system of grid. 
    returns: grid_gdf: grid cell geodataframe. 
    """
    # Calculate grid cell spacing in x and y directions
    width = np.diff(x_points)[0]
    height = np.diff(y_points)[0]
    
    # Create list specifying coordinates of grid cell centers
    coordinate_pairs = [(x,y) for x,y in product(x_points,y_points)]
    x_vals = [x for x,y in coordinate_pairs]
    y_vals = [y for x,y in coordinate_pairs]
    grid_cells = [box(x-width/2,y-height/2,x+width/2,y+height/2) for x,y in coordinate_pairs]
    grid_gdf = gpd.GeoDataFrame(data={'grid_x':x_vals,'grid_y':y_vals},geometry=grid_cells, crs=crs)
    
    return(grid_gdf)

def calculate_cell_weights(polygon,grid_gdf,poly_index):
    """
    param: polygon: shapely geometry specifying area over which to aggregate gridded data. Should be in a projected CRS. 
    param: grid_gdf: geopandas geodataframe of grid cell geometries. Should be in same projected CRS as polygon. 
    param: poly_index: unique identifier of polygon
    returns: overlap: pandas dataframe describing overlap between polygon and grid cells. 
    """
    overlap = grid_gdf.clip(polygon)
    overlap['overlap_area'] = overlap['geometry'].area
    overlap['poly_index'] = poly_index
    overlap.drop(columns='geometry',inplace=True)
    
    return(overlap)

def calculate_weightmap(polygon_gdf,grid_gdf):
    """
    param: polygon_gdf: geopandas geodataframe of polygons describing areas of interest (e.g., U.S. counties). Should be projected CRS. 
    param: grid_gdf: geopandas geodataframe of grid cell geometries. Should be in same projected CRS as polygon_gdf.
    returns: weightmap: pandas dataframe describing overlap between polygons and grid cells. 
    """
    
    # Get name of polygon index (defaults to poly_index)
    poly_index_name = polygon_gdf.index.name
    if poly_index_name is None:
        poly_index_name = 'poly_index'
    
    # For each polygon, get overlap with grid cells
    weightmap = pd.concat([calculate_cell_weights(polygon_gdf.loc[i,'geometry'],grid_gdf,i) for i in polygon_gdf.index.values])
    weightmap.rename(columns={'poly_index':poly_index_name},inplace=True)
    weightmap.reset_index(drop=True,inplace=True)
    
    return(weightmap)

### *** DATA INPUTS *** ###

pwd = os.getcwd()
gridded_crs = 'EPSG:4326'
projected_crs = 'EPSG:5070'
NHD_crs = 'EPSG:4269'

# Specify RPU of interest
RPU=sys.argv[1]
RPU_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/raster_processing/CONUS_raster_processing_units'
study_area = gpd.read_file(RPU_path).to_crs(projected_crs)
study_area = study_area[study_area['UnitID']==RPU]

buffer_degrees = 0.1
min_lon,min_lat,max_lon,max_lat = study_area.to_crs(gridded_crs).total_bounds

min_lon -= buffer_degrees
min_lat -= buffer_degrees
max_lon += buffer_degrees
max_lat += buffer_degrees

# Create output folder if it doesn't already exist 
outfolder = f'/proj/characklab/projects/kieranf/flood_damage_index/analysis/rainfall_runoff/aggregated_precip/{RPU}'
if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)
    
# NHDPlusV2 medium-resolution dataset (vector data) 
# Available at: https://www.epa.gov/waterdata/nhdplus-national-data
NHD_path = '/proj/characklab/projects/kieranf/flood_damage_index/data/NHDPlusMRData/NHDPlusNationalData/NHDPlusV21_National_Seamless_Flattened_Lower48.gdb'
NHD_mask = study_area.to_crs(NHD_crs)

catchments = gpd.read_file(NHD_path,layer='Catchment',mask=NHD_mask).to_crs(projected_crs).rename(columns={'FEATUREID':'COMID'})[['COMID','AreaSqKM','geometry']]

# NOAA Analysis of Record for Calibration (AORC) data
# Available at: https://registry.opendata.aws/noaa-nws-aorc/
AORC_dir = '/proj/characklab/projects/kieranf/flood_damage_index/data/AORC/'
year = 2024
gridded_data_path = os.path.join(AORC_dir,f'{year}.zarr')
gridded_data = xr.open_zarr(gridded_data_path)

### *** DETERMINE GRID CELL WEIGHTS FOR EACH CATCHMENT *** ###

# Create regular grid
lon_points = gridded_data['longitude'].to_numpy().round(8)
lat_points = gridded_data['latitude'].to_numpy().round(8)

lon_points = lon_points[(lon_points >= min_lon)&(lon_points <= max_lon)]
lat_points = lat_points[(lat_points >= min_lat)&(lat_points <= max_lat)]

grid_gdf = create_regular_grid(lon_points,lat_points,crs=gridded_crs).to_crs(projected_crs)

# Calculate weights based on overlap between grid cells and catchments
weightmap = calculate_weightmap(catchments.set_index('COMID'),grid_gdf)

# Convert to pyarrow data types
weightmap['overlap_area'] = weightmap['overlap_area'].round(0).astype('Int64[pyarrow]')
weightmap['grid_x'] = weightmap['grid_x'].astype('double[pyarrow]')
weightmap['grid_y'] = weightmap['grid_y'].astype('double[pyarrow]')
weightmap['COMID'] = weightmap['COMID'].astype('string[pyarrow]')

### *** SAVE RESULTS *** ###
outname = os.path.join(outfolder,f'{RPU}_AORC_weightmap.parquet')
weightmap.to_parquet(outname)