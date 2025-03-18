import numpy as np
import steady_state_diffusion_solver as ssd
import matplotlib.pyplot as plt
import rasterio as rio
import sys
import os

### *** HELPER FUNCTIONS *** ###

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

### *** INITIAL SETUP *** ###

# Specify HUC6 watershed of interest (passed as command-line argument)
huc6=sys.argv[1]

# Specify tolerance used to assess convergence of diffusion interpolation algorithm
tol=1e-5

# Specify value of over-relaxation parameter used by diffusion interpolation algorithm
omega=1.5

# Specify path to raster datasets
raster_dir = f'/proj/characklab/projects/kieranf/flood_damage_index/data/rasters/{huc6}'

### *** LOAD DATA *** ###

# Study area
with rio.open(os.path.join(raster_dir,f'{huc6}_study_area.tif'),'r') as src:
    study_area_raster = src.read(1)
    raster_crs = src.crs
    transform = src.transform
    out_shape = src.shape
    dx,dy = src.res
    
study_area_mask = (study_area_raster == 1)

# Elevation
with rio.open(os.path.join(raster_dir,f'{huc6}_elev.tif'),'r') as src:
    elev_raster = src.read(1)
    
# 100-year floodplain
with rio.open(os.path.join(raster_dir,f'{huc6}_floodplain_100y.tif'),'r') as src:
    floodplain_100y_raster = src.read(1)
    
# 500-year floodplain
with rio.open(os.path.join(raster_dir,f'{huc6}_floodplain_500y.tif'),'r') as src:
    floodplain_500y_raster = src.read(1)
    
# Initial guess for elevation of 100-year flood based on floodplain boundaries
with rio.open(os.path.join(raster_dir,f'{huc6}_wse_100y_guess.tif'),'r') as src:
    wse_100y_guess_raster = src.read(1)
    
# Initial guess for elevation of 500-year flood based on floodplain boundaries
with rio.open(os.path.join(raster_dir,f'{huc6}_wse_500y_guess.tif'),'r') as src:
    wse_500y_guess_raster = src.read(1)
    
### *** USE DIFFUSION INTERPOLATION TO CALCULATE ELEVATION OF 100-YEAR FLOODPLAIN *** ###

# Specify dirichlet boundary conditions as points bordering floodplain
dirichlet_bc = elev_raster.copy()

inside_mask = (floodplain_100y_raster==1)
outside_mask = (floodplain_100y_raster==0)
dirichlet_bc[inside_mask] = np.nan
dirichlet_bc[~study_area_mask] = -9999
nodata_mask = np.isnan(dirichlet_bc)
has_nodata_neighbor = get_neighbors(nodata_mask)

boundary_points = outside_mask&has_nodata_neighbor

dirichlet_bc[~boundary_points] = np.nan

# Specify area to use when assessing convergence
conv_mask = inside_mask

# Supply initial guess to solver, and iterate until steady state is reached 
wse_100y = ssd.steady_state_diffusion_2D(wse_100y_guess_raster,dirichlet_bc,dx=dx,dy=dy,conv_mask=conv_mask,tol=tol,omega=omega)

# Crop to area inside floodplain 
wse_100y[outside_mask] = np.nan
wse_100y *= study_area_raster

# Save to file
outname = os.path.join(raster_dir,f'{huc6}_wse_100y_inside.tif')
write_raster(wse_100y,outname,transform,raster_crs)

# Calculate height relative to 100-year floodplain 
h100y = elev_raster - wse_100y

outname = os.path.join(raster_dir,f'{huc6}_h100y_inside.tif')
write_raster(h100y,outname,transform,raster_crs)

### *** USE DIFFUSION INTERPOLATION TO CALCULATE ELEVATION OF 500-YEAR FLOODPLAIN *** ###

# Specify dirichlet boundary conditions as points bordering floodplain
dirichlet_bc = elev_raster.copy()

inside_mask = (floodplain_500y_raster==1)
outside_mask = (floodplain_500y_raster==0)
dirichlet_bc[inside_mask] = np.nan
dirichlet_bc[~study_area_mask] = -9999
nodata_mask = np.isnan(dirichlet_bc)
has_nodata_neighbor = get_neighbors(nodata_mask)

boundary_points = outside_mask&has_nodata_neighbor

dirichlet_bc[~boundary_points] = np.nan

# Specify area to use when assessing convergence
conv_mask = inside_mask

# Supply initial guess to solver, and iterate until steady state is reached 
wse_500y = ssd.steady_state_diffusion_2D(wse_500y_guess_raster,dirichlet_bc,dx=dx,dy=dy,conv_mask=conv_mask,tol=tol,omega=omega)

# Crop to area inside floodplain 
wse_500y[outside_mask] = np.nan
wse_500y *= study_area_raster

# Save to file
outname = os.path.join(raster_dir,f'{huc6}_wse_500y_inside.tif')
write_raster(wse_500y,outname,transform,raster_crs)

# Calculate height relative to 500-year floodplain 
h500y = elev_raster - wse_500y

outname = os.path.join(raster_dir,f'{huc6}_h500y_inside.tif')
write_raster(h500y,outname,transform,raster_crs)