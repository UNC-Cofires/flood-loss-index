import numpy as np
import pandas as pd
import os

### *** HELPER FUNCTIONS *** ###

def assign_presence_absence_points(polygon_id,num_presence,num_absence,building_lookup):
    """
    param: polygon_id: unique id of polygon of interest
    param: num_presence: number of presence points to sample within polygon
    param: num_absence: number of absence points to sample within polygon
    param: building_lookup: pandas groupby object listing buildings associated with each polygon
    """

    sampled_buildings = building_lookup.get_group(polygon_id).sample(num_presence+num_absence)
    flooded_status = np.zeros(len(sampled_buildings),dtype=int)
    flooded_inds = np.random.choice(np.arange(len(sampled_buildings)),size=num_presence,replace=False)
    flooded_status[flooded_inds] = 1
    sampled_buildings['flooded'] = flooded_status
    
    return(sampled_buildings)

def monte_carlo_sample_buildings(polygon_pa_counts,building_lookup,num_replicates=1):
    """
    param: polygon_pa_counts: pandas dataframe listing number of presence-absence points in each polygon, indexed by polygon_id
    param: building_lookup: pandas groupby object listing buildings associated with each polygon
    param: num_replicates: number of times to repeat monte carlo sampling
    """

    # Drop unnecessary entries before starting loop to save on computation time
    
    df_list = []
    
    for i in range(num_replicates):
        
        df = pd.concat([assign_presence_absence_points(polygon_id,row['num_presence'],row['num_absence'],building_lookup) for polygon_id,row in polygon_pa_counts.iterrows()])
        df.reset_index(inplace=True)
        df['replicate'] = i+1
        df_list.append(df)

    df = pd.concat(df_list).reset_index(drop=True)

    return(df)

### *** INITIAL SETUP *** ###

# Get current working directory 
pwd = os.getcwd()

# Get number of event to use for validation
event_number = int(os.environ['SLURM_ARRAY_TASK_ID'])

# Get event information
event_catalog_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/event_delineation/historical_TC_event_info.csv'
event_catalog = pd.read_csv(event_catalog_path)
event_catalog['START_DATE'] = pd.to_datetime(event_catalog['START_DATE'])
event_catalog['END_DATE'] = pd.to_datetime(event_catalog['END_DATE'])
event_info = event_catalog[event_catalog['EVENT_NUMBER']==event_number].iloc[0]
print(event_info,flush=True)

# Create folder for output 
outfolder = os.path.join(pwd,'presence_absence_data/stochastic_assignment/')
if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)

### *** LOAD DATA *** ###

## Presence-absence information derived from anonymized NFIP records
presence_absence_data_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/nfip_building_matching/presence_absence_data/presence_absence_summary.parquet'
filters = [('EVENT_NUMBER','=',event_number)]
presence_absence_data = pd.read_parquet(presence_absence_data_path,filters=filters).set_index('match_key')

## Building candidates
building_lookup_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/nfip_building_matching/potential_matches/CONUS_nfip_matching_building_lookup.parquet'
filters = [('match_key','in',presence_absence_data.index.values)]
building_lookup = pd.read_parquet(building_lookup_path,filters=filters)

# Group by match key so we can quickly look up buildings associated with each group
building_lookup = building_lookup.groupby('match_key',sort=False)

### *** ASSIGN RECORDS TO BUILDINGS *** ###

building_assignments = monte_carlo_sample_buildings(presence_absence_data,building_lookup,num_replicates=30)
building_assignments['EVENT_NUMBER'] = event_number
building_assignments = building_assignments[['EVENT_NUMBER','BUILD_ID','match_key','state','replicate','flooded']]

### *** SAVE RESULTS *** ###

outname = os.path.join(outfolder,f'event_{event_number:04d}_building_assignments.parquet')
building_assignments.to_parquet(outname)