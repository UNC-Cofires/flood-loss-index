import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import Affine
from rasterio.features import rasterize
from pyproj import CRS
import rioxarray
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import LinearNDInterpolator,NearestNDInterpolator
import matplotlib.pyplot as plt
import fiona
import pickle
import sys
import os

### *** HELPER FUNCTIONS *** ###

def read_pickled_geopandas(filepath):
    """
    Helper function to load a pickled geopandas geodataframe
    """
    
    with open(filepath,'rb') as f:
        gdf = pickle.load(f)
                
    return gdf

def rasterize_vector(gdf,out_shape,transform,burn_column=None,burn_value=1.0,fill_value=np.nan):
    """
    Function to rasterize vector geometries 
    
    param: geopandas geodataframe
    param: out_shape: tuple describing (width,height) of output raster
    param: transform: rasterio affine transform object describing locations of cells
    param: burn_column: column of geodataframe encoding value to be burned. If none, defaults to burn_value parameter.
    param: burn_value: default value to be burned if burn_column is None
    param: fill_value: fill value for unburned areas
    """

    if burn_column is not None:
        geom = [(shape,value) for shape,value in zip(gdf['geometry'],gdf[burn_column])]
    else:
        geom = [(shape,burn_value) for shape in gdf['geometry']]

    burned_raster = rasterize(geom,
                              out_shape = out_shape,
                              fill = fill_value,
                              out = None,
                              transform = transform,
                              all_touched = False,
                              dtype = float)
    
    return burned_raster

def write_raster(arr,filepath,transform,crs,nodata_value=np.nan):
    """
    Save a single-band raster to file
    
    param: arr: numpy array encoding raster values
    param: filepath: path to output file location
    param: transform: rasterio affine transform object describing locations of cells
    param: crs: pyproj crs object encoding coordinate reference system of raster
    param: nodata_value: number encoding missing values (e.g., np.nan, -9999)
    """
    
    with rio.open(
        filepath,
        mode="w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=arr.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata_value
    ) as dst:
        dst.write(arr, 1)
        
    return None

def get_neighbors(arr):
    """
    param: arr: 2D boolean array with points of interest labeled as "True"
    param: neighbor_mask: boolean array denoting elements that border a point of interest. 
    """
    padded = np.pad(arr,pad_width=1,mode='constant',constant_values=False)
    neighbor_mask = (padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:])
    return neighbor_mask

def interpolate_missing_values(arr,xx,yy,kind='linear',nodata_value=np.nan,region_mask=None): 
    """
    Function to interpolate NaN values in raster
    
    param: arr: 2D raster with missing values
    param: xx: x-coordinates of cells in arr
    param: yy: y-coordinates of cells in arr
    param: nodata_value: value denoting presence of missing data in arr
    param: region_mask: boolean array denoting area of interest (will ignore all elements that are false)
    """
    
    # Get locations of missing values
    if np.isnan(nodata_value):
        nodata_mask = np.isnan(arr)
    else:
        nodata_mask = (arr==nodata_value)
        
    known_mask = ~nodata_mask
    
    # To save time, don't use unecessary fixed/known points
    # Get only those that border a missing value
    has_nodata_neighbor = get_neighbors(nodata_mask)
    known_mask = known_mask&has_nodata_neighbor
    
    # Don't bother interpolating areas outside region of interest 
    if region_mask is None:
        region_mask = (np.ones(arr.shape)==1)
    
    known_mask = known_mask&region_mask
    nodata_mask = nodata_mask&region_mask
    
    known_points = list(zip(yy[known_mask],xx[known_mask]))
    known_values = arr[known_mask].flatten()
    
    if kind == 'linear':
        interp_func = LinearNDInterpolator(known_points, known_values)
    else:
        interp_func = NearestNDInterpolator(known_points, known_values)
        
    interp_values = interp_func(yy,xx)
    
    arr[nodata_mask] = interp_values[nodata_mask]
    
    return(arr)

### *** INITIAL SETUP *** ###

# Specify HUC6 watershed of interest (passed as command-line argument)
huc6=sys.argv[1]

# Create output folder if it doesn't already exist 
pwd = os.getcwd()
outfolder = f'/proj/characklab/projects/kieranf/flood_damage_index/data/rasters/{huc6}'
if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)

# Specify CRS
nhd_crs = 'EPSG:4269'   # Input (geographic) crs that we'll use to filter NHDPlusV2 data
conus_crs = 'EPSG:6350' # Output (projected) crs that we'll use for distance calculations

## Specify path to data sources
# Watershed boundary dataset
# Available at: https://www.usgs.gov/national-hydrography/access-national-hydrography-products
wbd_path = '/proj/characklab/projects/kieranf/flood_damage_index/data/watersheds/CONUS_WBD_HU6'

# NHDPlusV2 medium-resolution dataset
# Available at: https://www.epa.gov/waterdata/nhdplus-national-data
NHD_path = '/proj/characklab/projects/kieranf/flood_damage_index/data/NHDPlusMRData/NHDPlusNationalData/NHDPlusV21_National_Seamless_Flattened_Lower48.gdb'

# Enhanced NHDPlusV2 flow network dataset
# (This dataset more accurately models connectivity between flowlines than original release)
# Available at: https://doi.org/10.5066/P13IRYTB
ENHD_path = '/proj/characklab/projects/kieranf/flood_damage_index/data/ENHDPlusV2/enhd_nhdplusatts.parquet'

# FEMA National Flood Hazard Layer
# Available at: https://hazards.fema.gov/femaportal/NFHL/searchResult/
NFHL_path = '/proj/characklab/projects/kieranf/flood_damage_index/data/NFHL'

# Elevation and height above nearest drainage (HAND)
# Available at: https://cfim.ornl.gov/data/
ELEV_path = f'/proj/characklab/projects/kieranf/flood_damage_index/data/OakRidgeHAND/HU6_HAND/{huc6}/{huc6}.tif'
HAND_path = f'/proj/characklab/projects/kieranf/flood_damage_index/data/OakRidgeHAND/HU6_HAND/{huc6}/{huc6}hand.tif'

### *** LOAD DATA SOURCES *** ###

## Study area
wbd = gpd.read_file(wbd_path)
study_area = wbd[wbd['huc6']==huc6].to_crs(conus_crs)
study_area_states = study_area['states'].values[0].split(',')
study_area_mask_conus = study_area.buffer(1000).values[0]
study_area_mask_nhd = study_area.buffer(1000).to_crs(nhd_crs).values[0]

## NHD flowlines and catchment polygons

#This step can take a long time if you are reading in the entire CONUS
#To improve speed, use the "mask" argument to filter out data that falling outside study area
flowlines = gpd.read_file(NHD_path,layer='NHDFlowline_Network',mask=study_area_mask_nhd).to_crs(conus_crs)
catchments = gpd.read_file(NHD_path,layer='Catchment',mask=study_area_mask_nhd).to_crs(conus_crs).rename(columns={'FEATUREID':'COMID'})
nhd_areas = gpd.read_file(NHD_path,layer='NHDArea',mask=study_area_mask_nhd).to_crs(conus_crs)
nhd_waterbodies = gpd.read_file(NHD_path,layer='NHDWaterbody',mask=study_area_mask_nhd).to_crs(conus_crs)

# Get waterbodies that are coastal or part of flow network 
nhd_coastal_area_comids = nhd_areas[nhd_areas['FTYPE'].isin(['SeaOcean'])]['COMID'].unique()
nhd_coastal_areas = nhd_areas[nhd_areas['COMID'].isin(nhd_coastal_area_comids)]
nhd_areas = nhd_areas[nhd_areas['COMID'].isin(flowlines['WBAREACOMI'])]
nhd_waterbodies = nhd_waterbodies[nhd_waterbodies['COMID'].isin(flowlines['WBAREACOMI'])]

## Read in ENHD flowtable
flowtable = pd.read_parquet(ENHD_path)
flowtable[['comid','tocomid']] = flowtable[['comid','tocomid']].astype(int)

## FEMA NFHL
base_flood_elev = []
floodplain_100y = []
floodplain_500y = []

minx,miny,maxx,maxy = gpd.GeoSeries(study_area_mask_conus).total_bounds

for state in study_area_states:
    
    # Read in data on floodplain boundaries
    fld_dir = os.path.join(NFHL_path,state)
    fld_geom_path = os.path.join(fld_dir,f'{state}_FLD_HAZ_AR_geometry.pickle')
    fld_info_path = os.path.join(fld_dir,f'{state}_FLD_HAZ_AR.parquet')
    fld_geom = read_pickled_geopandas(fld_geom_path).cx[minx:maxx,miny:maxy]
    fld_info = pd.read_parquet(fld_info_path,columns=['FLD_ZONE','ZONE_SUBTY','SFHA_TF','STATIC_BFE','FLD_AR_ID'])
        
    fld_geom = pd.merge(fld_geom,fld_info,how='left',on='FLD_AR_ID')
    
    m_BFE = (fld_geom['STATIC_BFE'] > -9999)
    m_100y = (fld_geom['SFHA_TF']=='T')
    m_500y = m_100y|(fld_geom['ZONE_SUBTY']=='0.2 PCT ANNUAL CHANCE FLOOD HAZARD')
    
    base_flood_elev.append(fld_geom[m_BFE])
    floodplain_100y.append(fld_geom[m_100y])
    floodplain_500y.append(fld_geom[m_500y])
    
base_flood_elev = pd.concat(base_flood_elev)
floodplain_100y = pd.concat(floodplain_100y)
floodplain_500y = pd.concat(floodplain_500y)

### *** STUDY AREA RASTER *** ###

# Get bounds of HUC6 region that includes 1 km buffer
bbox = study_area.buffer(1000).total_bounds

# Specify cell size of rasters
cellsize = 90
upsize = 3*cellsize # Lowest resolution we anticipate upscaling to. Should be multiple of cellsize

# Round down to nearest 10m cell
bbox = (bbox // upsize)*upsize
minx,miny,maxx,maxy = bbox

# Get desired shape and dimensions of output raster
smallnum=1e-6
X = np.arange(minx+cellsize/2,maxx-cellsize/2+smallnum,cellsize)
Y = np.arange(miny+cellsize/2,maxy-cellsize/2+smallnum,cellsize)
XX,YY = np.meshgrid(X,Y)

transform = Affine.translation(X[0] - cellsize/2, Y[0] - cellsize/2) * Affine.scale(cellsize, cellsize)
raster_crs = CRS.from_string(conus_crs)
out_shape = (len(Y),len(X))

# Get mask raster denoting boundaries of watershed
study_area_raster = rasterize_vector(study_area,out_shape,transform,burn_value=1.0,fill_value=np.nan)
outname = os.path.join(outfolder,f'{huc6}_study_area.tif')
template_raster_path = outname # We'll later use this as a template when co-registering rasters 
write_raster(study_area_raster,outname,transform,raster_crs)

### *** ELEVATION RASTER *** ###

elev_array = rioxarray.open_rasterio(ELEV_path)
template_array = rioxarray.open_rasterio(template_raster_path)

# Co-register with study area raster
elev_array = elev_array.rio.reproject_match(template_array,resampling=Resampling.bilinear)

# Get underlying data as numpy array
elev_raster = elev_array.to_numpy()[0]
elev_raster[elev_raster==elev_array.rio.nodata] = np.nan

# Mask out areas that are water
land_mask = rasterize_vector(nhd_areas,out_shape,transform,burn_value=np.nan,fill_value=1.0)
land_mask *= rasterize_vector(nhd_coastal_areas,out_shape,transform,burn_value=np.nan,fill_value=1.0)
land_mask *= rasterize_vector(nhd_waterbodies,out_shape,transform,burn_value=np.nan,fill_value=1.0)
land_bool = np.where(np.isnan(land_mask), 0, land_mask)

elev_raster *= land_mask

## Fill holes 

# First pass with linear interpolation will fill most of it, but miss edges
elev_raster = interpolate_missing_values(elev_raster,XX,YY,kind='linear',region_mask=(study_area_raster==1))

# Second pass with nearest-neighbor will get remaining edges 
elev_raster = interpolate_missing_values(elev_raster,XX,YY,kind='nearest',region_mask=(study_area_raster==1))
elev_raster *= study_area_raster

outname = os.path.join(outfolder,f'{huc6}_elev.tif')
write_raster(elev_raster,outname,transform,raster_crs)

### *** HAND RASTER *** ###

hand_array = rioxarray.open_rasterio(HAND_path)

# Co-register with study area raster
hand_array = hand_array.rio.reproject_match(template_array,resampling=Resampling.bilinear)

# Get underlying data as numpy array
hand_raster = hand_array.to_numpy()[0]
hand_raster[hand_raster==hand_array.rio.nodata] = np.nan

hand_raster *= land_bool
hand_raster *= study_area_raster

outname = os.path.join(outfolder,f'{huc6}_hand.tif')
write_raster(hand_raster,outname,transform,raster_crs)

### *** WATERBODIES *** ###

waterbodies_raster = rasterize_vector(flowlines,out_shape,transform,burn_value=0.0,fill_value=1.0)
waterbodies_raster *= land_bool
waterbodies_raster = 1 - waterbodies_raster
waterbodies_raster *= study_area_raster

outname = os.path.join(outfolder,f'{huc6}_waterbodies.tif')
write_raster(waterbodies_raster,outname,transform,raster_crs)

### *** WATERBODIES (STREAM ORDER >=2) *** ###

waterbodies2_raster = rasterize_vector(flowlines[flowlines['StreamOrde']>=2],out_shape,transform,burn_value=0.0,fill_value=1.0)
waterbodies2_raster *= land_bool
waterbodies2_raster = 1 - waterbodies2_raster
waterbodies2_raster *= study_area_raster

outname = os.path.join(outfolder,f'{huc6}_waterbodies2.tif')
write_raster(waterbodies2_raster,outname,transform,raster_crs)

### *** 100-YEAR FLOODPLAIN *** ###

floodplain_100y_raster = rasterize_vector(floodplain_100y,out_shape,transform,burn_value=1.0,fill_value=0.0)
floodplain_100y_raster[land_bool==0]=1
floodplain_100y_raster *= study_area_raster

outname = os.path.join(outfolder,f'{huc6}_floodplain_100y.tif')
write_raster(floodplain_100y_raster,outname,transform,raster_crs)

### *** 500-YEAR FLOODPLAIN *** ###

floodplain_500y_raster = rasterize_vector(floodplain_500y,out_shape,transform,burn_value=1.0,fill_value=0.0)
floodplain_500y_raster[land_bool==0]=1
floodplain_500y_raster *= study_area_raster

outname = os.path.join(outfolder,f'{huc6}_floodplain_500y.tif')
write_raster(floodplain_500y_raster,outname,transform,raster_crs)

### *** ELEVATION OF 100-YEAR FLOODPLAIN *** ###

# Note: The metric doesn't necessarily correspond to the water surface elevation
# during a 1-in-X year flood (even though I'm conceptualizing it that way). 
# My approach is inpired by Zarekarizi et al. (2021), which I recommend reading. 
# Available at: https://doi.org/10.3390/w13050666
#
# This interpolation process is messy, and can create unphysical artifacts.
# However, because it's fast, we'll use it as the initial guess in script 
# that uses diffusion interpolation to produce a much smoother result. 

wse = elev_raster.copy()
wse_inside = elev_raster.copy()
wse_outside = elev_raster.copy()

inside_mask = (floodplain_100y_raster==1)
outside_mask = (floodplain_100y_raster==0)

wse_inside[inside_mask]=np.nan
wse_outside[outside_mask]=np.nan

# Use linear interpolation to evaluate elevation relative to floodplain boundaries within channel
wse_inside = interpolate_missing_values(wse_inside,XX,YY,kind='linear',region_mask=(study_area_raster==1))
wse_inside = interpolate_missing_values(wse_inside,XX,YY,kind='nearest',region_mask=(study_area_raster==1))

# Use nearest-neighbor interpolation to evaluate elevation relative to floodplain boundaries outside channel
wse_outside = interpolate_missing_values(wse_outside,XX,YY,kind='nearest',region_mask=(study_area_raster==1))

wse[inside_mask] = wse_inside[inside_mask]
wse[outside_mask] = wse_outside[outside_mask]

# Do final interpolation to provide initial guess in areas outside study region
# (Normally we don't want to do this, but it will speed up the diffusion interpolation step down the line)
wse = interpolate_missing_values(wse,XX,YY,kind='nearest')

outname = os.path.join(outfolder,f'{huc6}_wse_100y_guess.tif')
write_raster(wse,outname,transform,raster_crs)

### *** ELEVATION OF 500-YEAR FLOODPLAIN *** ###

wse = elev_raster.copy()
wse_inside = elev_raster.copy()
wse_outside = elev_raster.copy()

inside_mask = (floodplain_500y_raster==1)
outside_mask = (floodplain_500y_raster==0)

wse_inside[inside_mask]=np.nan
wse_outside[outside_mask]=np.nan

# Use linear interpolation to evaluate elevation relative to floodplain boundaries within channel
wse_inside = interpolate_missing_values(wse_inside,XX,YY,kind='linear',region_mask=(study_area_raster==1))
wse_inside = interpolate_missing_values(wse_inside,XX,YY,kind='nearest',region_mask=(study_area_raster==1))

# Use nearest-neighbor interpolation to evaluate elevation relative to floodplain boundaries outside channel
wse_outside = interpolate_missing_values(wse_outside,XX,YY,kind='nearest',region_mask=(study_area_raster==1))

wse[inside_mask] = wse_inside[inside_mask]
wse[outside_mask] = wse_outside[outside_mask]

# Do final interpolation to provide initial guess in areas outside study region
# (Normally we don't want to do this, but it will speed up the diffusion interpolation step down the line)
wse = interpolate_missing_values(wse,XX,YY,kind='nearest')

outname = os.path.join(outfolder,f'{huc6}_wse_500y_guess.tif')
write_raster(wse,outname,transform,raster_crs)

### *** NHD CATCHMENTS *** ###
catchments_raster = rasterize_vector(catchments,out_shape,transform,burn_column='COMID')
catchments_raster *= study_area_raster

outname = os.path.join(outfolder,f'{huc6}_nhd_catchments.tif')
write_raster(catchments_raster,outname,transform,raster_crs)

### *** NAN-VALUED VERSIONS OF SELECT RASTERS *** ###

# Certain GRASS GIS functions will prefer to have NAN values instead of zeros in binary rasters

waterbodies_raster[waterbodies_raster == 0] = np.nan
outname = os.path.join(outfolder,f'{huc6}_waterbodies_nanvalued.tif')
write_raster(waterbodies_raster,outname,transform,raster_crs)

waterbodies2_raster[waterbodies2_raster == 0] = np.nan
outname = os.path.join(outfolder,f'{huc6}_waterbodies2_nanvalued.tif')
write_raster(waterbodies2_raster,outname,transform,raster_crs)

floodplain_100y_raster[floodplain_100y_raster == 0] = np.nan
outname = os.path.join(outfolder,f'{huc6}_floodplain_100y_nanvalued.tif')
write_raster(floodplain_100y_raster,outname,transform,raster_crs)

floodplain_500y_raster[floodplain_500y_raster == 0] = np.nan
outname = os.path.join(outfolder,f'{huc6}_floodplain_500y_nanvalued.tif')
write_raster(floodplain_500y_raster,outname,transform,raster_crs)


