import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import os

### *** HELPER FUNCTIONS *** ###

def get_daily_max_zeta(filepath):
    """
    Get daily max water level at shoreline nodes included in the 
    Coastal Ocean Reanalysis (CORA) dataset. 

    https://registry.opendata.aws/noaa-nos-cora/
    
    param: filepath: path to CORA reanalysis data
    (NetCDF files in V1.1/assimilated/native_grid_shoreline_folder)
    """

    ds = xr.open_dataset(filepath)
    daily_max_zeta = ds['zeta'].resample(time='1D').max()

    t = daily_max_zeta['time'].values
    nodes = daily_max_zeta['nodenum'].values
    z = daily_max_zeta.to_numpy()
    
    df = pd.DataFrame(z,index=t,columns=nodes)
    
    return df

### *** INITIAL SETUP *** ###

# Get current working directory 
pwd = os.getcwd()

# Specify CRS
crs = 'EPSG:5070'

# Create output folder
outfolder = os.path.join(pwd,'CORA_extract')
if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)

# Get shoreline nodes
shoreline_filepath = '/proj/characklab/projects/kieranf/flood_damage_index/data/CORA/Shapefile_CORA_Shoreline'
shoreline_nodes = gpd.read_file(shoreline_filepath).to_crs(crs)
shoreline_nodes = shoreline_nodes[['node','geometry']].rename(columns={'node':'nodenum'})

# Get path to CORA files
cora_dir = '/proj/characklab/projects/kieranf/flood_damage_index/data/CORA/native_grid_shoreline'
cora_files = np.sort([x for x in os.listdir(cora_dir) if x.endswith('.nc')])
cora_filepaths = [os.path.join(cora_dir,file) for file in cora_files]

### *** EXTRACT DATA *** ###

# Get daily max water level at each node during 1979-2022 period
df = pd.concat([get_daily_max_zeta(path) for path in cora_filepaths],axis=1)
print(f'Share of nodes captured: {len(df.columns)} / {len(shoreline_nodes)} ({100*share_included:.2f}%)',flush=True)

# Save to file
outname = os.path.join(outfolder,'shoreline_nodes')
shoreline_nodes.to_file(outname)

outname = os.path.join(outfolder,'daily_max_zeta_at_shoreline_nodes.parquet')
df.to_parquet(outname)