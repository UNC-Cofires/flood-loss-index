import numpy as np
import pandas as pd
import networkx as nx
import pickle
import os
import gc

### *** HELPER FUNCTIONS *** ###

def antecedent_precipitation_index(P,k=0.9):
    """
    param: P: numpy array of hourly precipitation in period leading up to event (vector of length n). 
              The ith element of P should represent the total precip during the (n - i)th hour before
              the start of an event.
    param: k: decay constant (% per day). Typical values are between 0.8 and 1.0. 
    """
    n = len(P)
    t = np.flip(np.arange(n))
    API = np.sum(P*k**(t/24))

    return(API)

def summarize_precip(source_comid,G,catchment_area,precip,start_date,end_date,network=True):
    """
    param: source_comid: COMID of NHD catchment of interest
    param: G: graph representation of flow network. 
    param: catchment_area: dataframe listing area of each NHD catchment (indexed by COMID)
    param: precip: dataframe containing hourly precipitation data for each NHD catchment. Indexed by time. 
    param: start_date: start date of event. 
    param: end_date: end date of event. 
    param: network: flag denoting whether catchment is included in ENHD flow network (usually true)
    returns: res: dictionary containing various summary measures of event precipitation. 
    """

    # Flow wave travel times used to define "n-hour catchments" [hours]
    travel_time_intervals = [0,24,48,72]

    # Durations used to evaluate maximum precipitation intensity [hours]
    precip_duration_intervals = [3,6,12,24,48,72]

    # How far back to look when evaluating antecedent precipitation [hours]
    API_lookback_period = 120
    
    event_mask = (precip.index >= np.datetime64(start_date))&(precip.index <= np.datetime64(end_date))
    
    res = {'comid':source_comid}
    
    travel_time_cutoff = np.max(travel_time_intervals)
    
    # Trace flow upstream until we hit travel time cutoff
    upstream_dict = nx.single_source_dijkstra_path_length(G, source_comid, cutoff=travel_time_cutoff, weight='wave_travel_time_hr')
    upstream_df = pd.DataFrame({'comid':upstream_dict.keys(),'wave_travel_time_hr':upstream_dict.values()})
    upstream_df['areasqkm'] = catchment_area['areasqkm'].loc[upstream_df['comid']].values
        
    # Sometimes you can have NHD flowlines whose catchment area is zero;
    # these are often artificial paths or really short (<10m) stream segments. 
    # We don't have catchment-averaged precip for these COMIDs, so will drop them. 
    upstream_df = upstream_df[upstream_df['comid'].isin(precip.columns)]
    
    for travel_time in travel_time_intervals:
    
        m = (upstream_df['wave_travel_time_hr'] <= travel_time)
    
        # Get upstream NHD catchments that are reachable within specified travel time
        upstream_catchment_comids = upstream_df[m]['comid'].to_numpy()
        upstream_catchment_areas = upstream_df[m]['areasqkm'].to_numpy()
        upstream_catchment_precip = precip[upstream_catchment_comids].to_numpy()
    
        res[f'C{travel_time}_area_sqkm'] = upstream_catchment_areas.sum()
    
        # Calculate weighted average hourly precipitation over this region
        weighted_upstream_precip = np.sum(upstream_catchment_precip*upstream_catchment_areas,axis=1)/np.sum(upstream_catchment_areas)
        weighted_upstream_precip = pd.Series(weighted_upstream_precip,index=precip.index)
    
        # Calculate antecedent precipitation index
        API_values = weighted_upstream_precip.rolling(window=API_lookback_period).apply(antecedent_precipitation_index)
        API = API_values[event_mask].values[0]
        res[f'C{travel_time}_API{API_lookback_period}_mm'] = API
    
        # Calculate maximum average intensity (MAI) of rainfall for different durations
        for duration in precip_duration_intervals:
            res[f'C{travel_time}_MAI{duration}_mmhr'] = weighted_upstream_precip.rolling(window=duration).mean().max()

    return res

### *** INITIAL SETUP *** ###

# Get current working directory 
pwd = os.getcwd()

# Get event number
event_number = int(os.environ['SLURM_ARRAY_TASK_ID'])

# Create output folder
outfolder = os.path.join(pwd,'precip_by_event')
if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)

# Get event information
event_catalog_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/event_delineation/historical_TC_event_info.csv'
event_catalog = pd.read_csv(event_catalog_path)
event_catalog['START_DATE'] = pd.to_datetime(event_catalog['START_DATE'])
event_catalog['END_DATE'] = pd.to_datetime(event_catalog['END_DATE'])

event_info = event_catalog[event_catalog['EVENT_NUMBER']==event_number].iloc[0]
start_date = event_info['START_DATE']
end_date = event_info['END_DATE']

print(event_info,flush=True)

### *** LOAD DATA *** ###

## AORC precipitation (aggregated by NHD catchment)

precip_dir = os.path.join(pwd,'aggregated_precip')
RPU_list = np.sort(os.listdir(precip_dir))
buffer = pd.Timedelta(7,'days')
yearmonth_list = pd.date_range(start_date-buffer,end_date+buffer).strftime('%Y-%m').unique()

yearmonth_precip_list = []

for yearmonth in yearmonth_list:
    
    RPU_precip_filepaths = []
    
    for RPU in RPU_list:
        path = os.path.join(precip_dir,RPU,f'AORC_by_yearmonth/{RPU}_{yearmonth}_AORC_precip_by_COMID.parquet')
        RPU_precip_filepaths.append(path)

    yearmonth_precip = pd.concat([pd.read_parquet(filepath) for filepath in RPU_precip_filepaths],axis=1)
    yearmonth_precip_list.append(yearmonth_precip)
    
precip = pd.concat(yearmonth_precip_list,axis=0)

# Drop rows outside date range of interest
m = (precip.index >= np.datetime64(start_date-buffer))&(precip.index <= np.datetime64(end_date+buffer))
precip = precip[m]

# Can end up with duplicate columns (occurs when you have catchment touching RPU borders). 
# Drop duplicates so that each column is unique. 
precip = precip.loc[:,~precip.columns.duplicated()].copy()

# Cast columns as integers to be consistent with coding of COMIDs in flow network
precip.columns = precip.columns.astype(int)

# Free up memory
del yearmonth_precip_list
del yearmonth_precip
gc.collect()

## NHD flow network

# NHD catachment areas
catchment_area_path = os.path.join(pwd,'flow_network/nhd_catchment_area.parquet')
catchment_area = pd.read_parquet(catchment_area_path).set_index('comid')

# Graph representation of flow network (nx.DiGraph object with arrows pointing upstream) 
network_path = os.path.join(pwd,'flow_network/flow_network_directed_graph_upstream.pickle')
with open(network_path,'rb') as f:
    G = pickle.load(f)

### *** SUMMARIZE PRECIPITATION UPSTREAM OF EACH NHD CATCHMENT *** ###

comids = pd.Series(precip.columns)
event_precip_summary = pd.DataFrame([summarize_precip(comid,G,catchment_area,precip,start_date,end_date) for comid in comids])

### *** SAVE RESULTS *** ###

outname = os.path.join(outfolder,f'event_{event_number:04d}_precip_summary.parquet')
event_precip_summary.to_parquet(outname)