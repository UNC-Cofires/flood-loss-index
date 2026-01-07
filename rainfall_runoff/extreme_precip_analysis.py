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

    # Specify chunksize that xarray should use.
    # Since we are calculating annual maxima, we should chunk things 
    # up spatially and leave the time dimension intact. 
    chunks={'latitude':25,'longitude':25,'time':-1}
    
    # Read in AORC data from each year
    annual_maxima_list = []
    
    for year in np.arange(start_year,end_year+1):
        
        gridded_data_path = os.path.join(AORC_dir,f'{year}.zarr')

        if sample_points:
            gridded_data = xr.open_zarr(gridded_data_path).sel(latitude=lats,longitude=lons,method='nearest')
        elif sample_bbox:
            gridded_data = xr.open_zarr(gridded_data_path).sel(latitude=slice(min_lat,max_lat),longitude=slice(min_lon,max_lon)).chunk(chunks)
        else:
            gridded_data = xr.open_zarr(gridded_data_path).chunk(chunks)
        
        max_intensity = gridded_data['APCP_surface'].rolling(time=duration).mean().max(dim='time')
        max_intensity = max_intensity.expand_dims(dim='year').assign_coords(year=('year', [year]))
        max_intensity = max_intensity.astype('float32')
        
        annual_maxima_list.append(max_intensity)

    annual_maxima = xr.concat(annual_maxima_list,dim='year').chunk({'year':-1})
    
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

    # Calculate exceedence probability associated with return period of interest
    exceedance_prob = 1/return_period

    # Use inverse CDF transform to get value associated with 
    # exceedance probability
    extreme_value = distribution.ppf(1-exceedance_prob)

    return(extreme_value)

### *** CHARACTERIZE EXTREME VALUES *** ###

pwd = os.getcwd()

# Get values of command-line arguments 
RPU = sys.argv[1]
duration = int(sys.argv[2])            # Duration of interest [hours]
return_period = float(sys.argv[3])     # Return period of interest [years]

# Create folder for output
outfolder = os.path.join(pwd,f'extreme_values/{RPU}')
if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)

# Get bounding box of study area
RPU_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/raster_processing/CONUS_raster_processing_units'
study_area = gpd.read_file(RPU_path)
study_area = study_area[study_area['UnitID']==RPU].to_crs('EPSG:5070')
study_area['geometry'] = study_area['geometry'].buffer(10000) # Add 10 km buffer to be safe
study_area = study_area.to_crs('EPSG:4326')
bbox = tuple(study_area.total_bounds)

print(f'Creating timeseries of annual max {duration}-hour precipitation intensity.',flush=True)

# Path to NOAA Analysis of Record for Calibration (AORC) data
# Available at: https://registry.opendata.aws/noaa-nws-aorc/
AORC_dir = '/proj/characklab/projects/kieranf/flood_damage_index/data/AORC/'

# Extract timeseries of annual maximum precipitation intensity
annual_maxima = AORC_annual_max_precip_intensity(AORC_dir,duration=duration,bbox=bbox)

print(f'Estimating {duration}-hour precipitation intensity with return period of {return_period:.1f} years.',flush=True)

# Calculate precipitation intensity associated with return period of interest
rp_intensity = xr.apply_ufunc(extreme_value_analysis,
                              annual_maxima,
                              kwargs={'return_period':return_period},
                              input_core_dims=[['year']],
                              dask='parallelized',
                              vectorize=True,
                              output_dtypes=[np.float32])

### *** EXPORT RESULTS *** ###

# Rename lon/lat to x/y since this is what rasterio expects
rp_intensity = rp_intensity.rename({'longitude': 'x', 'latitude': 'y'})

# Specify CRS
rp_intensity = rp_intensity.rio.write_crs('EPSG:4326')

# Save to file
outname = os.path.join(outfolder,f'{RPU}_MAI{duration}_RP{return_period:.1f}y_mm_per_hr.tif')
rp_intensity.rio.to_raster(outname,tiled=True,windowed=True,compress='deflate')

print('Saved output to:',outname,flush=True)