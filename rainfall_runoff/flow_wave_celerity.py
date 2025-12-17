import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import pickle
import os

### *** INITIAL SETUP *** ###

# Get current working directory 
pwd = os.getcwd()

# Create output folder
outfolder = os.path.join(pwd,'flow_network')
if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)

### *** LOAD DATA *** ###

## NHDPlusV2 catchments
NHD_path = '/proj/characklab/projects/kieranf/flood_damage_index/data/NHDPlusMRData/NHDPlusNationalData/NHDPlusV21_National_Seamless_Flattened_Lower48.gdb'
catchments = gpd.read_file(NHD_path,layer='Catchment',columns=['FEATUREID','AreaSqKM'],ignore_geometry=True).rename(columns={'FEATUREID':'comid','AreaSqKM':'areasqkm'})

# Specify path to enhanced NHDPlusV2 flow network dataset
# (This dataset more accurately models connectivity between flowlines than original release)
# Available at: https://doi.org/10.5066/P13IRYTB
ENHD_path = '/proj/characklab/projects/kieranf/flood_damage_index/data/ENHDPlusV2/enhd_nhdplusatts.parquet'

## Read in ENHD flowtable
flowtable = pd.read_parquet(ENHD_path)
flowtable[['comid','tocomid']] = flowtable[['comid','tocomid']].astype(int)

## Read in channel geometry info from Zarrabi et al. (2025)
# (https://doi.org/10.1029/2024WR037997)

channel_geom_path = '/proj/characklab/projects/kieranf/flood_damage_index/data/channel_geometry/zarrabi_2025/Bankfull_Meanflow_CONUS.txt'

data_types = {'COMID':'int64[pyarrow]',
              'REACHCODE':'string[pyarrow]',
              'TotDASqKM':'double[pyarrow]',
              'StreamOrde':'int64[pyarrow]',
              'bnk_depth':'double[pyarrow]',
              'bnk_width':'double[pyarrow]',
              'mf_depth':'double[pyarrow]',
              'mf_width':'double[pyarrow]'}

channel_geom = pd.read_csv(channel_geom_path,index_col=0,dtype=data_types).rename(columns={'COMID':'comid'})

# Attach info on bankfull width/depth to flowtable
flowtable = pd.merge(flowtable,channel_geom[['comid','bnk_depth','bnk_width']],on='comid',how='left')

# For the small number of reaches with missing data, impute values based on
# median parameter value for reaches of same stream order
flowtable['streamorde'] = flowtable['streamorde'].astype('int64[pyarrow]')
median_bnk_width = flowtable.groupby('streamorde')['bnk_width'].median()
median_bnk_depth = flowtable.groupby('streamorde')['bnk_depth'].median()
median_roughness = flowtable.groupby('streamorde')['roughness'].median()

flowtable ['bnk_width']= flowtable.apply(lambda x: median_bnk_width[x['streamorde']] if pd.isna(x['bnk_width']) else x['bnk_width'],axis=1)
flowtable ['bnk_depth']= flowtable.apply(lambda x: median_bnk_depth[x['streamorde']] if pd.isna(x['bnk_depth']) else x['bnk_depth'],axis=1)
flowtable ['roughness']= flowtable.apply(lambda x: median_roughness[x['streamorde']] if pd.isna(x['roughness']) else x['roughness'],axis=1)

### *** CALCULATE TRAVEL TIMES *** ###

## Calculate flow wave celerity using Manning's formula
# Use same approach as Allen et al. (2018) (https://doi.org/10.1029%2F2018GL077914)
# Assumptions: Rectangular channel, kinematic wave approximation

# Set minimum slope value to avoid zero-slope problems
min_slope = 1e-5
flowtable['slope'] = np.maximum(flowtable['slope'],min_slope)

# Hydraulic ratius (assumes rectangular cross-section) [m]
Rh = flowtable['bnk_width']*flowtable['bnk_depth'] / (flowtable['bnk_width'] + 2*flowtable['bnk_depth'])

# Slope [m/m]
S = flowtable['slope'] 

# Gauckler–Manning coefficient [s/m^(1/3)]
n = flowtable['roughness']

# Cross-sectional average velocity [m/s]
u = (1/n)*Rh**(2/3)*S**(1/2)

# Flow wave celerity (assumes rectangular channel) [m/s]
c = (5/3)*u

# Reach length [m]
L = 1000*flowtable['lengthkm']

# Flow wave travel time [s]
travel_time = L/c

flowtable['flow_velocity_m_per_sec'] = u
flowtable['wave_celerity_m_per_sec'] = c
flowtable['wave_travel_time_sec'] = travel_time
flowtable['wave_travel_time_hr'] = travel_time/3600

### *** GRAPH REPRESENTATION OF FLOW NETWORK *** ###

# Represent flow network as directed graph whose edge weights are based on wave travel time

# Initialize directed graph object
G_downstream = nx.DiGraph()

# Add nodes
node_info = pd.concat([flowtable[['comid','areasqkm']],catchments[['comid','areasqkm']]]).drop_duplicates(subset=['comid']).reset_index(drop=True)
nodes = [(int(x[0]),{'areasqkm':x[1]}) for x in node_info.to_numpy()]
G_downstream.add_nodes_from(nodes)

# Add edges
terminal_flows = (flowtable['tocomid']==0) # Want to exclude terminal flows 
edges = [(int(x[0]),int(x[1]),{'lengthkm':x[2],'wave_travel_time_hr':x[3]}) for x in flowtable[~terminal_flows][['comid','tocomid','lengthkm','wave_travel_time_hr']].to_numpy()]
G_downstream.add_edges_from(edges)

# Create version of graph with arrows reversed that will allow us to look upstream
G_upstream = G_downstream.reverse(copy=True)

### *** SAVE RESULTS *** ###

outname = os.path.join(outfolder,'enhd_flowtable_modified.parquet')
flowtable.to_parquet(outname)

outname = os.path.join(outfolder,'nhd_catchment_area.parquet')
node_info.to_parquet(outname)

outname = os.path.join(outfolder,'flow_network_directed_graph_downstream.pickle')
with open(outname, 'wb') as f:
    pickle.dump(G_downstream, f)

outname = os.path.join(outfolder,'flow_network_directed_graph_upstream.pickle')
with open(outname, 'wb') as f:
    pickle.dump(G_upstream, f)