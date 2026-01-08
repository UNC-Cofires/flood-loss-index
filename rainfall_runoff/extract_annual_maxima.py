import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
from dask.diagnostics import ProgressBar
import sys
import os

### *** HELPER FUNCTIONS *** ###

def AORC_annual_max_precip_intensity(AORC_dir,duration=24,start_year=1979,end_year=2024,bbox=None):
    """
    This function calculates the annual maximum precipitation intensity [mm/hr] for a given 
    duration (e.g., 24 hours) based on NOAA's Analysis of Record for Calibration (AORC) dataset. 
    
    param: AORC_dir: path to AORC data folder. This folder will contain a series of 
                    zarr files subsetted by year. 
    param: duration: number of hours in rolling window used to calculate precipitation intensity
    param: start_year: first year to include in calculation. 
    param: end_year: last year to include in calculation. 
    param: bbox: bounding box describing area within which to sample data. This should be a tuple 
                    with the following structure: (min_lon,min_lat,max_lon,max_lat). 
    returns: annual_maxima: xarray dataset of annual max precipitation intensity indexed by 
                    year and latitude/longitude. Units of mm/hr. Note that intensities are 
                    calculated over a rolling window whose size is controlled by the duration 
                    parameter (e.g., duration=24 corresponds to annual max 24-hour intensity). 
    """        

    # Build file list
    filepaths = [os.path.join(AORC_dir, f"{year}.zarr") for year in np.arange(start_year, end_year + 1)]

    # Specify chunksize that xarray should use.
    # Time chunksize should be larger than duration of interest.
    chunks={'latitude':128,'longitude':256,'time':720}

    # Open files as an xarray dataset but don't read them into memory yet
    gridded_data = xr.open_mfdataset(filepaths,engine="zarr",concat_dim="time",combine="nested",chunks=chunks)

    # Subset to region of interest
    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        gridded_data = gridded_data.sel(latitude=slice(min_lat, max_lat),longitude=slice(min_lon, max_lon))

    # Get hourly precipitation variable and cast as float32 to save memory 
    precip = gridded_data['APCP_surface'].astype('float32')

    # Rolling n-hour precipitation intensity
    rolling_precip = precip.rolling(time=duration).sum() / duration

    # Get annual max n-hour precipitation intensity
    annual_max_intensity = rolling_precip.groupby("time.year").max("time")
    
    return annual_max_intensity

### *** INITIAL SETUP *** ###

pwd = os.getcwd()

# Get values of command-line arguments 
duration = int(sys.argv[1])                       # Duration of interest [hours]
task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])  # Study domain chunk number to focus on

print(f'SLURM_ARRAY_TASK_ID: {task_id}',flush=True)

# Specify path to NOAA Analysis of Record for Calibration (AORC) data
# Available at: https://registry.opendata.aws/noaa-nws-aorc/
AORC_dir = '/proj/characklab/projects/kieranf/flood_damage_index/data/AORC/'

# Create folder for output
outfolder = os.path.join(pwd,f'extreme_values/AORC_annual_maxima')
if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)

outname = os.path.join(outfolder,f'AORC_annual_max_{duration}hr_precip_intensity.zarr')

### *** GET PROPERTIES OF AORC GRID *** ###

AORC_template = xr.open_zarr(os.path.join(AORC_dir,'1979.zarr'))
AORC_lats = AORC_template['latitude'].to_numpy()
AORC_lons = AORC_template['longitude'].to_numpy()
years = np.arange(1979,2024+1)

# Create zarr store for output if it doesn't already exist

if not os.path.exists(outname):
    template_arr = np.empty((len(AORC_lats),len(AORC_lons),len(years)),dtype='float32')
    coords = {'latitude':AORC_lats,'longitude':AORC_lons,'year':years}
    chunks = {'latitude':128,'longitude':256,'year':-1}
    template_data = xr.DataArray(template_arr,coords,name='APCP_surface').chunk(chunks)
    template_data.to_zarr(outname,mode="w",zarr_version=2,consolidated=True)

# Figure out what region this particular SLURM job should focus on 

lat_chunksize = 128*5
lon_chunksize = 256*5
num_lat_chunks = np.ceil(len(AORC_lats)/lat_chunksize).astype(int)
num_lon_chunks = np.ceil(len(AORC_lons)/lon_chunksize).astype(int)
num_chunks = num_lat_chunks*num_lon_chunks
lat_chunk = task_id // num_lon_chunks
lon_chunk = task_id % num_lon_chunks

lat_idx = lat_chunk*lat_chunksize
lon_idx = lon_chunk*lon_chunksize

lat_slice = slice(lat_idx,lat_idx+lat_chunksize)
lon_slice = slice(lon_idx,lon_idx+lon_chunksize)

min_lon = AORC_lons[lon_slice].min()
min_lat = AORC_lats[lat_slice].min()
max_lon = AORC_lons[lon_slice].max()
max_lat = AORC_lats[lat_slice].max()

# Get bbox for geographic filtering of data
bbox = (min_lon,min_lat,max_lon,max_lat)
bbox_str = '(' + ','.join([f'{x:.4f}' for x in bbox]) + ')'

# Get indices of output zarr that this corresponds to
region = {'latitude':lat_slice,'longitude':lon_slice,'year':slice(None)}

print(f'\nFocusing on AORC domain chunk {task_id + 1} / {num_chunks}: {bbox_str}',flush=True)

### *** EXTRACT ANNUAL MAXIMA *** ###

print(f'\nCreating timeseries of annual max {duration}-hour precipitation intensity.',flush=True)

# Extract timeseries of annual maximum precipitation intensity
annual_maxima = AORC_annual_max_precip_intensity(AORC_dir,duration=duration,bbox=bbox)

with ProgressBar(dt=10):
    annual_maxima = annual_maxima.persist()

### *** SAVE RESULTS *** ###

print(f'\nRechunking to ensure zarr compatibility.',flush=True)

with ProgressBar(dt=10):
    annual_maxima = annual_maxima.chunk({'latitude':128,'longitude':256,'year':-1})

print(f'\nWriting output to {outname}',flush=True)

with ProgressBar(dt=10):
    annual_maxima.to_zarr(outname,zarr_version=2,region=region,mode='r+',consolidated=False)

print(f'\nTask complete.',flush=True)