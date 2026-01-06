import numpy as np
import pandas as pd
import geopandas as gpd
import scipy.stats as stats
import xarray as xr
import rioxarray
import sys
import os

### *** HELPER FUNCTIONS *** ###

def AORC_annual_max_precip_intensity(AORC_dir,duration=24,start_year=1979,end_year=2024,bbox=None,points=None):
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
    param: points: geopandas geodataframe of points at which to sample data.
                    If not specified, precipitation intensity will be calculated over
                    the entire AORC domain. 
    returns: annual_maxima: xarray dataset of annual max precipitation intensity indexed by 
                    year and latitude/longitude. Units of mm/hr. Note that intensities are 
                    calculated over a rolling window whose size is controlled by the duration 
                    parameter (e.g., duration=24 corresponds to annual max 24-hour intensity). 
    """        

    # Determine if user wants to sample data at points, within a bounding box, or across the entire domain. 
    
    if points is not None:
        sample_points = True
        sample_bbox = False
        points = points.to_crs('EPSG:4326')
        lons = xr.DataArray(points.geometry.x.to_numpy(), dims='points')
        lats = xr.DataArray(points.geometry.y.to_numpy(), dims='points')

    if bbox is not None:
        min_lon,min_lat,max_lon,max_lat = bbox
        sample_points = False
        sample_bbox = True

    else:
        sample_points = False
        sample_bbox = False

    # Read in AORC data from each year
    
    annual_maxima_list = []
    
    for year in np.arange(start_year,end_year+1):
        
        gridded_data_path = os.path.join(AORC_dir,f'{year}.zarr')

        if sample_points:
            gridded_data = xr.open_zarr(gridded_data_path).sel(latitude=lats,longitude=lons,method='nearest')
        elif sample_bbox:
            gridded_data = xr.open_zarr(gridded_data_path).sel(latitude=slice(min_lat,max_lat),longitude=slice(min_lon,max_lon))
        else:
            gridded_data = xr.open_zarr(gridded_data_path)
        
        max_intensity = gridded_data['APCP_surface'].rolling(time=duration).mean().max(dim='time')
        max_intensity = max_intensity.expand_dims(dim='year').assign_coords(year=('year', [year]))
        annual_maxima_list.append(max_intensity)

    annual_maxima = xr.concat(annual_maxima_list,dim='year')
    
    return annual_maxima

def extreme_value_analysis(annual_maxima,return_period=2.0,c_guess=0.0):
    """
    This function estimates the precipitation intensity associated with a user-specified
    return period using the annual maximum series (AMS) approach. Annual maxima are 
    assumed to follow a generalized extreme value (GEV) distribution. For more information 
    on the parameters of this distribution, please see the following scipy documentation

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html
    
    param: annual_maxima: timeseries of annual maximum values [1-D numpy array].
    param: return_period: return period of interest (e.g., 1-year, 2-year, 10-year, etc.) [float].
    param: c_guess: initial guess for shape parameter c [float]. 
    """

    # Fit GEV distribution to timeseries of annual maxima
    c, loc, scale = stats.genextreme.fit(annual_maxima,c_guess)
    distribution = stats.genextreme(c=c,loc=loc,scale=scale)

    # Calculate exceedence probability associated with return period
    exceedance_prob = 1/return_period

    # Use inverse CDF transform to get value associated with 
    # exceedance probability
    extreme_value = distribution.ppf(1-exceedance_prob)

    return(extreme_value)

### *** CHARACTERIZE EXTREME VALUES *** ###

pwd = os.getcwd()

# Create folder for output
outfolder = os.path.join(pwd,'extreme_values')
if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)

# Get duration and return period of interest
# (these values are passed by the user as command-line arguments)
duration = int(sys.argv[1])
return_period = float(sys.argv[2])

print(f'Calculating maximum average {duration}-hour precipitation intensity with return period of {return_period:.1f} years.',flush=True)

# Path to NOAA Analysis of Record for Calibration (AORC) data
# Available at: https://registry.opendata.aws/noaa-nws-aorc/
AORC_dir = '/proj/characklab/projects/kieranf/flood_damage_index/data/AORC/'

# Extract timeseries of annual maximum precipitation intensity
annual_maxima = AORC_annual_max_precip_intensity(AORC_dir,duration=duration)

# Assemble into local memory 
annual_maxima = annual_maxima.compute()

# Calculate precipitation intensity associated with return period of interest
rp_intensity = xr.apply_ufunc(extreme_value_analysis,
                              annual_maxima,
                              kwargs={'return_period':return_period},
                              input_core_dims=[['year']], vectorize=True)

### *** EXPORT RESULTS *** ###

# Cast as float32 to save memory
rp_intensity = rp_intensity.astype('float32')

# Rename lon/lat to x/y since this is what rasterio expects
rp_intensity = rp_intensity.rename({'longitude': 'x', 'latitude': 'y'})

# Sort rows of raster so that they are oriented north-to-south
if rp_intensity.y[0] < rp_intensity.y[-1]:
    rp_intensity = rp_intensity.sortby('y', ascending=False)

# Specify CRS
rp_intensity = rp_intensity.rio.write_crs('EPSG:4326')

# Save to file
outname = os.path.join(outfolder,f'MAI{duration}_RP{return_period:.1f}y_mm_per_hr.tif')
rp_intensity.rio.to_raster(outname)

print('Saved output to:',outname,flush=True)