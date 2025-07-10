import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import sys
import os

### *** HELPER FUNCTIONS *** ##
def create_coordinate_keys(df,x_col,y_col,decimal_places=6):
    """
    This function creates a string representation of coordinate points that can be used as a key for merging data. 
    (We do this because merging on a float key often does not work properly due to machine precision issues)
    """    
    x_str = df[x_col].apply(lambda x: f'{x:.{decimal_places}f}')
    y_str = df[y_col].apply(lambda y: f'{y:.{decimal_places}f}')
    point_str = '(' + x_str + ',' + y_str + ')'
    return point_str

def weighted_average_precip(precip,timepoints,weightmap,catchment_col,weight_col,point_col):
    """
    param: precip: numpy array of precipitation timeseries (n_timepoints x n_points)
    param: timepoints: numpy array listing dates of observation (length n_timepoints) 
    param: weightmap: pandas dataframe describing overlap between catchment polygons and gridded precipitation data. 
    param: catchment_col: column of weightmap denoting catchment_id
    param: weight_col: column of weightmap denoting weight associated with each precipitation point. 
    param: point_col: column of weightmap denoting index of precipitation point in precip array. 
    """
    
    # Index weightmap by catchment (if not done so already) 
    if weightmap.index.name != catchment_col:
        weightmap = weightmap.set_index(catchment_col)
    
    # Get list of catchment_ids
    catchment_id_list = np.unique(weightmap.index.values)
    n_catchments = len(catchment_id_list)
    
    # Preallocate array for weighted averate precipitation data
    n_timepoints = len(timepoints)
    weighted_precip = np.zeros((n_timepoints,n_catchments))
    
    for j,catchment_id in enumerate(catchment_id_list):
        
        point_indices = weightmap.loc[[catchment_id],point_col].to_numpy()
        point_weights = weightmap.loc[[catchment_id],weight_col].to_numpy()
        weighted_precip[:,j] = np.sum(precip[:,point_indices]*point_weights,axis=1)/np.sum(point_weights)
        
    return pd.DataFrame(data=weighted_precip,columns=catchment_id_list,index=timepoints)

### *** DATA INPUTS *** ###

# Read in command-line arguments
RPU = sys.argv[1]
min_year = int(sys.argv[2])
max_year = int(sys.argv[3])

# Set up folder 
pwd = os.getcwd()
aggregation_folder = os.path.join(pwd,f'aggregated_precip/{RPU}')

# Create folder for output
outfolder = os.path.join(aggregation_folder,f'AORC_by_yearmonth')
if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)

# Weightmap 
weightmap_path = os.path.join(aggregation_folder,f'{RPU}_AORC_weightmap.parquet')
weightmap = pd.read_parquet(weightmap_path)
weightmap['point_str'] = create_coordinate_keys(weightmap,'grid_x','grid_y')

# Get unique coordinate points associated with grid
coordinates = weightmap[['grid_x','grid_y']].drop_duplicates(ignore_index=True)
coordinates['point_index'] = coordinates.index.values
coordinates['point_str'] = create_coordinate_keys(coordinates,'grid_x','grid_y')

# Attach coordinate index to weightmap
weightmap = pd.merge(weightmap,coordinates[['point_index','point_str']],on='point_str',how='left')

# Index weightmap by catchment COMID
weightmap.set_index('COMID',inplace=True)

# NOAA Analysis of Record for Calibration (AORC) data
# Available at: https://registry.opendata.aws/noaa-nws-aorc/
AORC_dir = '/proj/characklab/projects/kieranf/flood_damage_index/data/AORC/'
AORC_precip_var = 'APCP_surface'

# Specify points of interest to read in
x_coords = xr.DataArray(coordinates['grid_x'].to_list(), dims="points")
y_coords = xr.DataArray(coordinates['grid_y'].to_list(), dims="points")

### *** CALCULATE WEIGHTED-AVERAGE RAINFALL OVER NHD CATCHMENTS *** ###

# For each year, aggregate precip 
period_list = pd.period_range(start=f'{min_year}-01', end=f'{max_year}-12',freq='M')

print(f'Calculating weighted average precip for {min_year}-{max_year} period',flush=True)
    
for period in period_list:
    
    print(f'    {str(period)}',flush=True)
        
    # Get start and end dates of period
    period_str = str(period)
    period_start = np.datetime64(period.start_time)
    period_end = np.datetime64(period.end_time)
    year = period.year
    
    # Read in AORC data 
    gridded_data_path = os.path.join(AORC_dir,f'{year}.zarr')
    gridded_data = xr.open_zarr(gridded_data_path).sel(time=slice(period_start,period_end))
    gridded_data = gridded_data[AORC_precip_var].sel(longitude=x_coords, latitude=y_coords, method='nearest')
    timepoints = gridded_data.time.to_numpy()
    precip = gridded_data.to_numpy()

    # Fill nan values
    np.nan_to_num(precip,nan=0.0,copy=False)
    
    # Aggregate precipitation timeseries over NHD catchments
    catchment_precip = weighted_average_precip(precip,timepoints,weightmap,'COMID','overlap_area','point_index')
    
    # Save results
    outname = os.path.join(outfolder,f'{RPU}_{period_str}_AORC_precip_by_COMID.parquet')
    catchment_precip.to_parquet(outname)