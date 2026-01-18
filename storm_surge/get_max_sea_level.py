import numpy as np
import pandas as pd
import geopandas as gpd
import os

### *** HELPER FUNCTIONS *** ###
def get_zmax_by_event(zeta,event_info):
    """
    param: zeta: pandas dataframe of daily max water levels at each station (indexed by date)
    param: event_info: row of pandas dataframe providing information on specific TC event
                       (must contain EVENT_NUMBER, START_DATE, END_DATE columns)

    returns: event_zmax_df: pandas dataframe describing max water level at each station during the event
    """

    event_mask = (zeta.index >= event_info['START_DATE'])&(zeta.index <= event_info['END_DATE'])
    event_zmax = zeta[event_mask].max()
    event_zmax_df = pd.DataFrame(data={'EVENT_NUMBER':event_info['EVENT_NUMBER'],'nodenum':event_zmax.index.values,'zmax':event_zmax.values})

    return event_zmax_df

### *** MAIN SCRIPT *** ###

# Get current working directory 
pwd = os.getcwd()

# Read in data on historical TC events
# (Exclude those from after 2022 since that's when the CORA data stops)
event_catalog_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/event_delineation/historical_TC_event_info.csv'
event_catalog = pd.read_csv(event_catalog_path)
event_catalog['START_DATE'] = pd.to_datetime(event_catalog['START_DATE']).dt.tz_localize(None)
event_catalog['END_DATE'] = pd.to_datetime(event_catalog['END_DATE']).dt.tz_localize(None)
event_catalog = event_catalog[event_catalog['SEASON'] <= 2022]

# Read in data on daily max water levels during 1979-2022 period
daily_max_zeta_path = os.path.join(pwd,'CORA_extract/daily_max_zeta_at_shoreline_nodes.parquet')
daily_max_zeta = pd.read_parquet(daily_max_zeta_path)

# A small number of values are unrealistically large (>100m)
# Replace these with NAs
daily_max_zeta.mask(daily_max_zeta > 100, pd.NA, inplace=True)

# Read in data on shoreline nodes
shoreline_nodes_path = os.path.join(pwd,'CORA_extract/shoreline_nodes')
shoreline_nodes = gpd.read_file(shoreline_nodes_path)
num_nodes = len(shoreline_nodes)

# Drop nodes that don't show up in data (should be very small)
m = shoreline_nodes['nodenum'].isin(daily_max_zeta.columns)
shoreline_nodes = shoreline_nodes[m].reset_index(drop=True)

# Start list of nodes that we'll exclude due to missing data
nodes_to_drop = shoreline_nodes[~m]['nodenum'].to_list()

# Get max elevation over threshold for each event
zmax_by_event = pd.concat([get_zmax_by_event(daily_max_zeta,event_info) for i,event_info in event_catalog.iterrows()])

# Get list of nodes with missing data
nodes_to_drop += list(set(zmax_by_event[zmax_by_event['zmax'].isna()]['nodenum'].unique()))
nodes_to_drop = np.unique(nodes_to_drop)

# Exclude nodes with missing data
zmax_by_event = zmax_by_event[~zmax_by_event['nodenum'].isin(nodes_to_drop)].reset_index(drop=True)
shoreline_nodes = shoreline_nodes[~shoreline_nodes['nodenum'].isin(nodes_to_drop)].reset_index(drop=True)

print(f'Dropped {len(nodes_to_drop)} / {num_nodes} nodes with missing data.',flush=True)

# Get the 95th percentile of daily water levels

p95 = daily_max_zeta.quantile(0.95)
p95 = pd.DataFrame(p95).reset_index()
p95.columns=['nodenum','P95_daily_zmax']

zmax_by_event = pd.merge(zmax_by_event,p95,on='nodenum',how='left')

# Save results

outfolder = os.path.join(pwd,'CORA_max_zeta_by_event')

if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)

outname = os.path.join(outfolder,'max_zeta_by_event.parquet')
zmax_by_event.to_parquet(outname)

outname = os.path.join(outfolder,'included_shoreline_nodes')
shoreline_nodes.to_file(outname)