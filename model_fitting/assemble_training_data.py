import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import os

### *** HELPER FUNCTIONS *** ###

def downcast_floats(df,columns=None):
    """
    Helper function to convert float columns of dataframes from float64 to float32 to
    save on memory. 

    param: columns: subset of columns to apply function to. If none, will apply to all float columns in df. 
    """
    if columns is None:
        columns = list(df.columns)

    column_dtypes = df[columns].dtypes.astype('string')

    for col in column_dtypes[column_dtypes.str.contains('float64')].index.values:
        df[col] = df[col].astype('float32')

    return df

def harmonize_dtypes(df,columns=None):
    """
    Helper function to convert string and integer columns of dataframes to consistent data types. 

    param: columns: subset of columns to apply function to. If none, will apply to all float columns in df. 
    """

    if columns is None:
        columns = list(df.columns)

    column_dtypes = df[columns].dtypes.astype('string')
    
    integer_column_mask = (column_dtypes.str.contains('int'))
    string_column_mask = (column_dtypes.str.contains('string'))|(column_dtypes.str.contains('str'))|(column_dtypes.str.contains('object'))

    for col in column_dtypes[integer_column_mask].index.values:
        df[col] = df[col].astype('Int64')

    for col in column_dtypes[string_column_mask].index.values:
        df[col] = df[col].astype('string')

    return df

def extract_latlon_from_match_key(match_key):
    
    components = match_key.split('&')
    lat = [s for s in components if 'latitude' in s][0]
    lon = [s for s in components if 'longitude' in s][0]
    lat = lat.strip('()').split('==')[-1]
    lon = lon.strip('()').split('==')[-1]
    latlon_coord = '(' + lat + ',' + lon + ')'
    
    return latlon_coord

### *** INITIAL SETUP *** ###

# Get current working directory 
pwd = os.getcwd()

# Create output folders
disagg_outfolder = os.path.join(pwd,'training_data/disaggregated')
if not os.path.exists(disagg_outfolder):
    os.makedirs(disagg_outfolder,exist_ok=True)

agg_outfolder = os.path.join(pwd,'training_data/aggregated')
if not os.path.exists(agg_outfolder):
    os.makedirs(agg_outfolder,exist_ok=True)

# Get event information
event_number = int(os.environ['SLURM_ARRAY_TASK_ID'])
event_catalog_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/event_delineation/historical_TC_event_info.csv'
event_catalog = pd.read_csv(event_catalog_path)
event_catalog['START_DATE'] = pd.to_datetime(event_catalog['START_DATE'])
event_catalog['END_DATE'] = pd.to_datetime(event_catalog['END_DATE'])
event_info = event_catalog[event_catalog['EVENT_NUMBER']==event_number].iloc[0]

print(event_info,flush=True)

### *** LOAD DATA *** ###

## Presence-absence information
presence_absence_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/nfip_building_matching/presence_absence_data/presence_absence_summary.parquet'
filters = [('EVENT_NUMBER','=',event_number)]
presence_absence_data = pd.read_parquet(presence_absence_path,filters=filters)

# Calculate claim rate
presence_absence_data['num_records'] = presence_absence_data['num_presence'] + presence_absence_data['num_absence']
presence_absence_data['claim_intensity'] = presence_absence_data['num_presence'] / presence_absence_data['num_records']

# Extract latitude/longitude from match keys
presence_absence_data['SPATIAL_BLOCK_ID'] = presence_absence_data['match_key'].apply(extract_latlon_from_match_key)
included_match_keys = presence_absence_data['match_key'].unique().tolist()

# Harmonize data types
presence_absence_data = harmonize_dtypes(presence_absence_data)

## Building candidates

building_lookup_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/nfip_building_matching/potential_matches/CONUS_nfip_matching_building_lookup.parquet'
filters = [('match_key','in',included_match_keys)]
building_lookup = dd.read_parquet(building_lookup_path,filters=filters)
building_lookup = harmonize_dtypes(building_lookup)

## Topographic features

included_RPUs = ['03a','03b','03c','03d','03e','03f'] # (!) expand once finalized. 
raster_dir = '/proj/characklab/projects/kieranf/flood_damage_index/data/rasters'
raster_filepaths = [os.path.join(raster_dir,f'{RPU}/{RPU}_raster_values_at_structure_points.parquet') for RPU in included_RPUs]
topo_data = dd.concat([dd.read_parquet(filepath) for filepath in raster_filepaths]).reset_index(drop=True)
topo_data['nhd_catchment_comid'] = topo_data['nhd_catchment_comid'].astype('int64[pyarrow]')
topo_data['cora_shoreline_node'] = topo_data['cora_shoreline_node'].astype('int64[pyarrow]')

topo_features = ['dist_coast_m',
                 'dist_wb_m',
                 'elev_cm',
                 'hand_wb_cm']

topo_data = topo_data[['BUILD_ID','nhd_catchment_comid','cora_shoreline_node']+topo_features]
topo_data = downcast_floats(topo_data)
topo_data = harmonize_dtypes(topo_data)

## Precipitation intensity

precip_filepath = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/rainfall_runoff/precip_by_event/combined_precip_by_event.parquet'
filters = [('EVENT_NUMBER','=',event_number)]
precip_data = dd.read_parquet(precip_filepath,filters=filters).rename(columns={'comid':'nhd_catchment_comid'})

precip_features = ['C0_area_sqkm','C0_API120_mm','C0_MAI24_mmhr']

precip_data = precip_data[['EVENT_NUMBER','nhd_catchment_comid']+precip_features]
precip_data = downcast_floats(precip_data)
precip_data = harmonize_dtypes(precip_data)

## Storm surge

storm_surge_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/storm_surge/CORA_max_zeta_by_event/max_zeta_by_event.parquet'
filters = [('EVENT_NUMBER','=',event_number)]
storm_surge_data = dd.read_parquet(storm_surge_path,filters=filters).rename(columns={'nodenum':'cora_shoreline_node'})

storm_surge_features = ['zmax','P95_daily_zmax']

storm_surge_data = storm_surge_data[['EVENT_NUMBER','cora_shoreline_node']+storm_surge_features]
storm_surge_data = downcast_floats(storm_surge_data)
storm_surge_data = harmonize_dtypes(storm_surge_data)

### *** MERGE DATA *** ###

# Attach data on structure attributes (static)
presence_absence_data = dd.merge(presence_absence_data,building_lookup,on='match_key',how='left')

# Attach data on topographic attributes (static)
presence_absence_data = dd.merge(presence_absence_data,topo_data,on='BUILD_ID',how='left')

# Attach data on precipitation intensity (dynamic) 
presence_absence_data = dd.merge(presence_absence_data,precip_data,on=['EVENT_NUMBER','nhd_catchment_comid'],how='left')

# Attach data on storm surge conditions (dynamic) 
presence_absence_data = dd.merge(presence_absence_data,storm_surge_data,on=['EVENT_NUMBER','cora_shoreline_node'],how='left')

# Assemble dataset into local memory as a pandas dataframe
print('Assembling data into local memory.',flush=True)
with ProgressBar(dt=1):
    presence_absence_data = presence_absence_data.compute()

### *** CALCULATE AVERAGE VALUE OF FEATURES WITHIN GROUPS *** ###

# Get list of features to use in model
features = topo_features + precip_features + storm_surge_features 

agg_cols = ['EVENT_NUMBER','match_key']

agg_dict = {'SPATIAL_BLOCK_ID':'first',
            'num_claims':'first',
            'num_policies':'first',
            'num_buildings':'first',
            'num_presence':'first',
            'num_absence':'first',
            'num_records':'first',
            'claim_intensity':'first'}

for feature in features:
    agg_dict[feature] = 'mean'

agg_data = presence_absence_data.groupby(agg_cols).agg(agg_dict).reset_index()

### *** SAVE RESULTS *** ###

# Downcast to float32 to save space
presence_absence_data = downcast_floats(presence_absence_data)
presence_absence_data = harmonize_dtypes(presence_absence_data)
agg_data = downcast_floats(agg_data)
agg_data = harmonize_dtypes(agg_data)

# Save to parquet file
disagg_outname = os.path.join(disagg_outfolder,f'event_{event_number:04d}_disaggregated_training_data.parquet')
presence_absence_data.to_parquet(disagg_outname)

agg_outname = os.path.join(agg_outfolder,f'event_{event_number:04d}_aggregated_training_data.parquet')
agg_data.to_parquet(agg_outname)

print(f'Saved output to {disagg_outname}\n',flush=True)
print(f'Saved output to {agg_outname}\n',flush=True)
