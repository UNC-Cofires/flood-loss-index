import numpy as np
import rasterio as rio
from rasterio.features import rasterize
from rastran import flowpath_integral,sample_nearest
from scipy.ndimage import distance_transform_edt
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

def calculate_hand(study_area,elev,fdr,wb_input,dx,dy,nodata_value,maxiter=1000,cutoff=0):
    """
    Calculate stream distance and height above nearest drainage (HAND)
    
    param: study_area: study area raster. Should be equal to 1 within study area, and nodata otherwise. 
    param: elev: elevation raster. 
    param: fdr: D8 flow direction raster. Follows GRASS GIS labeling convention.
    param: wb_input: raster denoting location of waterbodies. Should be equal to 1 within waterbodies, and nodata otherwise. 
    param: dx: raster resolution in x-direction (meters)
    param: dy: raster resolution in y-direction (meters)
    param: nodata_value: value used within rasters to denote NULL or nodata values. 
    param: maxiter: maximum number of steps to take when tracing flowpath back to nearest waterbody. 
    param: cutoff: cutoff distance at which flow paths are "snapped" to nearest stream based on euclidean distance. 
    returns: dist_sdt: stream distance raster (meters). 
    returns: hand_raster: height above nearest drainage (HAND) raster. 
    """
    
    # Get raster attributes
    study_area_mask = (study_area != nodata_value)
    nrows,ncols = study_area.shape
    
    # Create coordinate grid
    ii,jj = np.meshgrid(np.arange(nrows),np.arange(ncols),indexing='ij')
    
    # Compute stepsize raster (will integrate this to get travel distance) 
    i_step = reclass_raster(fdr,{1:-1,2:-1,3:-1,4:0,5:+1,6:+1,7:+1,8:0})
    j_step = reclass_raster(fdr,{1:+1,2:0,3:-1,4:-1,5:-1,6:0,7:+1,8:+1})

    stepsize = np.sqrt((i_step*dy)**2+(j_step*dx)**2)
    stepsize[i_step==nodata_value] = nodata_value
    stepsize[j_step==nodata_value] = nodata_value

    integrand = np.ones(stepsize.shape)
    
    # Set distance to waterbodies equal to zero on waterbodies
    wb_input[wb_input==1]=0
    
    # Calculate euclidean distance to waterbodies
    # We do this as a pre-processing step because we often get better results will fewer missing values
    # when flowpaths are "snapped" to streams once they come within a certain radius (typically a small value)
    dist_edt,indices = distance_transform_edt(wb_input,sampling=(dy,dx),return_distances=True, return_indices=True)

    i_nearest_edt = indices[0]
    j_nearest_edt = indices[1]

    dist_sdt = dist_edt.copy()
    i_nearest_sdt = i_nearest_edt.copy()
    j_nearest_sdt = j_nearest_edt.copy()
    
    # Create a buffer around streams that will be used to "snap" flowpaths to waterbodies
    # once they come within the cutoff distance
    # To disable this behaviour, set cutoff equal to zero
    stream_buffer_mask = (dist_edt <= cutoff)

    dist_sdt[~stream_buffer_mask] = nodata_value
    i_nearest_sdt[~stream_buffer_mask] = nodata_value
    j_nearest_sdt[~stream_buffer_mask] = nodata_value
    
    # Get list of coordinates to calculate stream distance for, and format as fortran array
    flow_distance_mask = (~stream_buffer_mask)&(study_area_mask)
    i_coords = ii[flow_distance_mask]
    j_coords = jj[flow_distance_mask]

    i_nearest_sdt = np.asfortranarray(i_nearest_sdt)
    j_nearest_sdt = np.asfortranarray(j_nearest_sdt)
    dist_sdt = np.asfortranarray(dist_sdt)
    
    # Compute distance to nearest stream using complied fortran subroutine
    flowpath_integral(i_coords,j_coords,i_step,j_step,stepsize,integrand,nodata_value,maxiter,i_nearest_sdt,j_nearest_sdt,dist_sdt)
    
    # Compute height above nearest stream
    elev_nearest_stream = sample_nearest(nodata_value,i_nearest_sdt,j_nearest_sdt,elev)
    
    # Set nodata values
    nodata_mask = (elev_nearest_stream == nodata_value)|(elev == nodata_value)

    hand_raster = elev - elev_nearest_stream
    hand_raster[nodata_mask] = nodata_value

    hand_raster = hand_raster.astype(elev.dtype)
    
    return(dist_sdt,hand_raster)

### *** LOAD DATA SOURCES *** ###

# Specify raster processing unit (RPU) of interest (passed as command-line argument)
RPU=sys.argv[1] 
outfolder = f'/proj/characklab/projects/kieranf/flood_damage_index/data/rasters/{RPU}'

# Specify path to raster data files
study_area_path = os.path.join(outfolder,f'{RPU}_study_area.tif')
elev_path = os.path.join(outfolder,f'{RPU}_elev_cm.tif')
fdr_path = os.path.join(outfolder,f'{RPU}_fdr.tif')
wb_path = os.path.join(outfolder,f'{RPU}_wb.tif')
wb2_path = os.path.join(outfolder,f'{RPU}_wb2.tif')

## Read in rasters 

# Get information on projection, resolution, and data type from study area raster
with rio.open(study_area_path,'r') as src:
    study_area = src.read(1)
    raster_crs = src.crs
    transform = src.transform
    out_shape = src.shape
    dx,dy = src.res
    nodata_value = src.nodata
    
# Elevation
with rio.open(elev_path,'r') as src:
    elev = src.read(1)

# Flow direction
with rio.open(fdr_path,'r') as src:
    fdr = src.read(1)

# Waterbodies
with rio.open(wb_path,'r') as src:
    wb = src.read(1)

# Waterbodies (stream order >= 2)
with rio.open(wb2_path,'r') as src:
    wb2 = src.read(1)
    
### *** CALCULATE HAND AND STREAM DISTANCE *** ###

# Calculate distance to nearest stream and height above nearest drainage
# (do this once for all waterbodies, and once for all waterbodies with a stream order >= 2)
dist_wb,hand_wb = calculate_hand(study_area,elev,fdr,wb,dx,dy,nodata_value,cutoff=90)
dist_wb2,hand_wb2 = calculate_hand(study_area,elev,fdr,wb2,dx,dy,nodata_value,cutoff=90)

# Convert to Int32 data type
dist_wb = dist_wb.astype(elev.dtype)
dist_wb2 = dist_wb2.astype(elev.dtype)

### *** SAVE RESULTS *** ###

# HAND rasters
outname = os.path.join(outfolder,f'{RPU}_hand_wb_cm.tif')
write_raster(hand_wb,outname,transform,raster_crs,nodata_value=nodata_value)

outname = os.path.join(outfolder,f'{RPU}_hand_wb2_cm.tif')
write_raster(hand_wb2,outname,transform,raster_crs,nodata_value=nodata_value)

# Stream distance rasters
outname = os.path.join(outfolder,f'{RPU}_dist_wb_m.tif')
write_raster(dist_wb,outname,transform,raster_crs,nodata_value=nodata_value)

outname = os.path.join(outfolder,f'{RPU}_dist_wb2_m.tif')
write_raster(dist_wb2,outname,transform,raster_crs,nodata_value=nodata_value)