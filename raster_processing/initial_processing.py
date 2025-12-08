import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt,binary_propagation
import sys
import os

### *** HELPER FUNCTIONS *** ###

def reclass_raster(arr,rules):
    """
    Creates a raster whose category values are based upon 
    a reclassification of the categories in an existing raster.
    
    param: arr: numpy array representing input raster
    param: rules: dictionary encoding oldvalue:newvalue pairs
    returns: arr_reclass: reclassified output raster
    """
    arr_reclass = arr.copy()
    
    for oldvalue,newvalue in rules.items():
        arr_reclass[arr==oldvalue] = newvalue
        
    return(arr_reclass)

def rasterize_vector(gdf,out_shape,transform,burn_column=None,burn_value=1.0,fill_value=np.nan,dtype=float):
    """
    Function to rasterize vector geometries 
    
    param: geopandas geodataframe
    param: out_shape: tuple describing (width,height) of output raster
    param: transform: rasterio affine transform object describing locations of cells
    param: burn_column: column of geodataframe encoding value to be burned. If none, defaults to burn_value parameter.
    param: burn_value: default value to be burned if burn_column is None
    param: fill_value: fill value for unburned areas
    param: dtype: data chosen data type
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
                              dtype = dtype)
    
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

### *** INITIAL SETUP *** ###

# Specify raster processing unit (RPU) of interest (passed as command-line argument)
RPU=sys.argv[1]

# Specify path to RPU shapefile
pwd = os.getcwd()
RPU_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/raster_processing/CONUS_raster_processing_units'

# Create output folder if it doesn't already exist 
outfolder = f'/proj/characklab/projects/kieranf/flood_damage_index/data/rasters/{RPU}'
if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)
    
## Specify path to data sources

# NHDPlusV2 medium-resolution dataset (vector data) 
# Available at: https://www.epa.gov/waterdata/nhdplus-national-data
NHD_path = '/proj/characklab/projects/kieranf/flood_damage_index/data/NHDPlusMRData/NHDPlusNationalData/NHDPlusV21_National_Seamless_Flattened_Lower48.gdb'
NHD_crs = 'EPSG:4269'

# Enhanced NHDPlusV2 flow network dataset
# (This dataset more accurately models connectivity between flowlines than original release)
# Available at: https://doi.org/10.5066/P13IRYTB
ENHD_path = '/proj/characklab/projects/kieranf/flood_damage_index/data/ENHDPlusV2/enhd_nhdplusatts.parquet'

# NHDPlusV2 elevation, flow direction, and flow accumulation rasters for RPU of interest
NHD_raster_dir = '/proj/characklab/projects/kieranf/flood_damage_index/data/NHDPlusMRData/NHDPlusRasters'
elev_path = os.path.join(NHD_raster_dir,f'{RPU}/{RPU}_elev_cm.tif')
fdr_path = os.path.join(NHD_raster_dir,f'{RPU}/{RPU}_fdr.tif')
fac_path = os.path.join(NHD_raster_dir,f'{RPU}/{RPU}_fac.tif')

# Coastal Ocean Reanalysis (CORA) shoreline nodes
shoreline_node_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/storm_surge/CORA_max_zeta_by_event/included_shoreline_nodes'

### *** LOAD DATA SOURCES *** ###

# RPU boundary
study_area = gpd.read_file(RPU_path)
study_area = study_area[study_area['UnitID']==RPU]

# Specify desired data type and nodata value for rasters
# (Note that we can reduce disk usage by saving things as integers) 
raster_dtype = np.int32
nodata_value = -999999

## Elevation raster (will use as template for other rasters) 
with rio.open(elev_path,'r') as src:
    elev_raster = src.read(1)
    raster_crs = src.crs
    transform = src.transform
    out_shape = src.shape
    dx,dy = src.res
    original_nodata_value = src.nodata
    
# Print EPSG code
print('EPSG:',raster_crs.to_epsg())

# Update nodata value
elev_raster = elev_raster.astype(raster_dtype)
elev_raster[elev_raster==original_nodata_value] = nodata_value

# Convert study area geodataframe to raster CRS,
# and create mask we can use to filter NHD data
study_area = study_area.to_crs(raster_crs)
buffered_study_area = study_area.copy()
buffered_study_area['geometry'] = buffered_study_area['geometry'].buffer(10000)
nhd_mask = buffered_study_area.to_crs(NHD_crs).geometry.values[0]

## Flowlines and catchments 
#This step can take a long time if you are reading in the entire CONUS
#To improve speed, use the "mask" argument to filter out data that falling outside study area
flowlines = gpd.read_file(NHD_path,layer='NHDFlowline_Network',mask=nhd_mask).to_crs(raster_crs)
catchments = gpd.read_file(NHD_path,layer='Catchment',mask=nhd_mask).to_crs(raster_crs).rename(columns={'FEATUREID':'COMID'})
nhd_areas = gpd.read_file(NHD_path,layer='NHDArea',mask=nhd_mask).to_crs(raster_crs)
nhd_waterbodies = gpd.read_file(NHD_path,layer='NHDWaterbody',mask=nhd_mask).to_crs(raster_crs)

# Get waterbodies that are coastal or part of flow network 
nhd_coastal_area_comids = nhd_areas[nhd_areas['FTYPE'].isin(['SeaOcean'])]['COMID'].unique()
nhd_coastal_areas = nhd_areas[nhd_areas['COMID'].isin(nhd_coastal_area_comids)]
nhd_areas = nhd_areas[nhd_areas['COMID'].isin(flowlines['WBAREACOMI'])]
nhd_waterbodies = nhd_waterbodies[nhd_waterbodies['COMID'].isin(flowlines['WBAREACOMI'])]

## CORA shoreline nodes
shoreline_nodes = gpd.read_file(shoreline_node_path).to_crs(raster_crs)
shoreline_nodes = shoreline_nodes[shoreline_nodes['geometry'].intersects(buffered_study_area['geometry'].values[0])]

### *** STUDY AREA RASTER *** ###

study_area_raster = rasterize_vector(study_area,out_shape,transform,burn_value=1,fill_value=nodata_value,dtype=raster_dtype)

study_area_mask = (study_area_raster==1)
nodata_mask = (study_area_raster==nodata_value)

### *** WATERBODIES RASTER *** ###

# Mask out areas that are water
land_mask = rasterize_vector(nhd_areas,out_shape,transform,burn_value=np.nan,fill_value=1.0)
land_mask *= rasterize_vector(nhd_coastal_areas,out_shape,transform,burn_value=np.nan,fill_value=1.0)
land_mask *= rasterize_vector(nhd_waterbodies,out_shape,transform,burn_value=np.nan,fill_value=1.0)
land_bool = np.where(np.isnan(land_mask), 0, land_mask)

# Add in flowlines 
waterbodies_raster = rasterize_vector(flowlines,out_shape,transform,burn_value=0.0,fill_value=1.0)
waterbodies_raster *= land_bool
waterbodies_raster = 1 - waterbodies_raster

# Convert to desired data type
waterbodies_raster = waterbodies_raster.astype(raster_dtype)

# Set areas that aren't waterbody to nodata (certain GRASS GIS functions require this) 
waterbodies_raster[waterbodies_raster==0] = nodata_value

### *** WATERBODIES RASTER (STREAM ORDER >= 2) *** ###

# Add in flowlines 
waterbodies2_raster = rasterize_vector(flowlines[flowlines['StreamOrde']>=2],out_shape,transform,burn_value=0.0,fill_value=1.0)
waterbodies2_raster *= land_bool
waterbodies2_raster = 1 - waterbodies2_raster

# Convert to desired data type
waterbodies2_raster = waterbodies2_raster.astype(raster_dtype)

# Set areas that aren't waterbody to nodata (certain GRASS GIS functions require this) 
waterbodies2_raster[waterbodies2_raster==0] = nodata_value

### *** NHD CATCHMENTS *** ###

catchments_raster = rasterize_vector(catchments,out_shape,transform,burn_column='COMID',fill_value=nodata_value,dtype=raster_dtype)
catchments_raster[nodata_mask] = nodata_value

### *** FLOW DIRECTION RASTER *** ###

with rio.open(fdr_path,'r') as src:
    fdr_raster = src.read(1)
    original_nodata_value = src.nodata
    
# Double check that shape matches that of other rasters
if fdr_raster.shape != elev_raster.shape:
    raise ValueError('Shape of rasters does not match.')
    
fdr_raster = fdr_raster.astype(raster_dtype)
fdr_raster[fdr_raster==original_nodata_value] = nodata_value

# Existing FDR raster uses ESRI convention for flow direction labels.
# Reclassify to be consistent with convention used by GRASS. 
rules = {1:8,2:7,4:6,8:5,16:4,32:3,64:2,128:1}
fdr_raster = reclass_raster(fdr_raster,rules)

### *** FLOW ACCUMULATION RASTER *** ###

with rio.open(fac_path,'r') as src:
    fac_raster = src.read(1)
    original_nodata_value = src.nodata
    
# Double check that shape matches that of other rasters
if fac_raster.shape != elev_raster.shape:
    raise ValueError('Shape of rasters does not match.')
    
fac_raster = fac_raster.astype(raster_dtype)
fac_raster[fac_raster==original_nodata_value] = nodata_value

### *** SHORELINE RASTER *** ###

if len(shoreline_nodes) > 0:

    coastal_areas_raster = rasterize_vector(nhd_coastal_areas,out_shape,transform,burn_value=1,fill_value=0,dtype=np.int32)

    # Create raster with locations and IDs of CORA shoreline nodes 
    shoreline_nodeloc_raster = rasterize_vector(shoreline_nodes,out_shape,transform,burn_value=1,fill_value=0,dtype=np.int32)
    shoreline_nodenum_raster = rasterize_vector(shoreline_nodes,out_shape,transform,burn_column='nodenum',fill_value=nodata_value,dtype=np.int32)
    
    # Determine nearest shoreline node for each grid cell
    dist_node_edt,dist_node_indices = distance_transform_edt(1-shoreline_nodeloc_raster,sampling=(dy,dx),return_distances=True, return_indices=True)

    # Delineate "territory" of each shoreline node
    i_nearest_node = dist_node_indices[0]
    j_nearest_node = dist_node_indices[1]
    nearest_node_raster = shoreline_nodenum_raster[i_nearest_node,j_nearest_node]
    
    # Identify coastline based on locations of shoreline nodes and user-specified criteria

    # Allow to propagate fully in any area labeled by NHD as sea or ocean
    coastal_mask = (coastal_areas_raster == 1)

    # Otherwise, allow to propagate within other waterbodies subject to certain conditions
    m1 = (elev_raster < 10*100)           # Restrict to areas <10m above sea level (note that elevation raster is in cm) 
    m2 = (elev_raster == nodata_value)    # Also include areas where elevation is undefined
    m3 = (waterbodies_raster == 1)        # Restrict to areas covered by NHD waterbodies
    m4 = (dist_node_edt < 5000)           # Restrict to areas within 5 km of a CORA shoreline node
    allowed_noncoastal_mask = (m1|m2)&m3&m4

    mask = (coastal_mask|allowed_noncoastal_mask).astype(int)
    
    # Create continous coastline by expanding outwards from shoreline nodes
    shoreline_raster = binary_propagation(shoreline_nodeloc_raster, mask=mask).astype(np.int32)

    # Compute distance to coast
    dist_coast_raster = distance_transform_edt(1-shoreline_raster,sampling=(dy,dx),return_distances=True)

    # Convert to integer data types
    dist_coast_raster[np.isnan(dist_coast_raster)] = nodata_value
    dist_coast_raster = dist_coast_raster.astype(np.int32)
    nearest_node_raster[np.isnan(nearest_node_raster)] = nodata_value
    nearest_node_raster = nearest_node_raster.astype(np.int32)

else:
    dist_coast_raster = nodata_value*np.ones(out_shape,dtype=np.int32)
    nearest_node_raster = nodata_value*np.ones(out_shape,dtype=np.int32)


### *** SAVE RESULTS *** ###

# Study area raster
outname = os.path.join(outfolder,f'{RPU}_study_area.tif')
write_raster(study_area_raster,outname,transform,raster_crs,nodata_value=nodata_value)

# Elevation raster
outname = os.path.join(outfolder,f'{RPU}_elev_cm.tif')
write_raster(elev_raster,outname,transform,raster_crs,nodata_value=nodata_value)

# Flow direction raster
outname = os.path.join(outfolder,f'{RPU}_fdr.tif')
write_raster(fdr_raster,outname,transform,raster_crs,nodata_value=nodata_value)

# Flow accumulation raster
outname = os.path.join(outfolder,f'{RPU}_fac.tif')
write_raster(fac_raster,outname,transform,raster_crs,nodata_value=nodata_value)

# Waterbodies
outname = os.path.join(outfolder,f'{RPU}_wb.tif')
write_raster(waterbodies_raster,outname,transform,raster_crs,nodata_value=nodata_value)

# Waterbodies (Stream Order >= 2)
outname = os.path.join(outfolder,f'{RPU}_wb2.tif')
write_raster(waterbodies2_raster,outname,transform,raster_crs,nodata_value=nodata_value)

# NHD Catchments
outname = os.path.join(outfolder,f'{RPU}_nhd_catchment_comid.tif')
write_raster(catchments_raster,outname,transform,raster_crs,nodata_value=nodata_value)

# Distance to coast
outname = os.path.join(outfolder,f'{RPU}_dist_coast_m.tif')
write_raster(dist_coast_raster,outname,transform,raster_crs,nodata_value=nodata_value)

# CORA shoreline nodes
outname = os.path.join(outfolder,f'{RPU}_cora_shoreline_node.tif')
write_raster(nearest_node_raster,outname,transform,raster_crs,nodata_value=nodata_value)
