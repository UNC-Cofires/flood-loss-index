import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os

### *** HELPER FUNCTIONS *** ###

def is_sfha_zone(x):
    """
    This function determines if the flood zone code (e.g., AE) listed in OpenFEMA corresponds to the SFHA.
    Returns 1 for SFHA zones, and zero otherwise. 
    """
    if not pd.isna(x):
        return int(x.startswith('A') or x.startswith('V'))
    else:
        return 0

def is_coastal_flood_zone(x):
    """
    This function determines if the flood zone code (e.g., VE, X, AE, etc.) listed in OpenFEMA corresponds
    a coastal flood hazard zone. 
    """
    if not pd.isna(x):
        return int(x.startswith('V'))
    else:
        return 0

def simplify_openfema_occupancy_types(openfema_occtype):
    """
    This function collapses the occupancy types listed in the OpenFEMA NFIP claims & policies datasets
    into the following simplified categories: 

    (1) residential_single_family
    (2) residential_two_to_four_family
    (3) residential_other
    (4) non_residential
    
    param: openfema_occtype: integer denoting occupancy type listed in OpenFEMA dataset (e.g., 11)
    """

    simplified_occtype = pd.NA

    if openfema_occtype in [1,11]:
        simplified_occtype = 'residential_single_family'
    elif openfema_occtype in [2,12]:
        simplified_occtype = 'residential_two_to_four_family'
    elif openfema_occtype in [3,13,14,15,16]:
        simplified_occtype = 'residential_other'
    elif openfema_occtype in [4,6,17,18,19]:
        simplified_occtype = 'non_residential'

    return simplified_occtype


def find_matching_records(left,right,matching_cols,multiple_value_cols=[],preallocate_per_record=1500,preallocate_cutoff=3000):
    """
    This function identifies potential matches between OpenFEMA records and NSI building points. 
    
    param: left: pandas dataframe of claim/policy records. Index of each row should be unique.
    param: right: pandas dataframe of building points. Index of each row should be unique. 
    param: matching_cols: list of columns used to match records in the left dataframe to records 
                          in the right dataframe. Each column should be present and represented 
                          as a consistent data type in both the left and right dataframes. If a 
                          field can take on multiple values in the right dataframe, then the 
                          column should have "_values" appended to its name in the right 
                          dataframe but not the left dataframe. 
    param: multiple_value_cols: list of columns in the right dataframe that are allowed to take on
                                multiple values (e.g., a building can have multiple 
                                censusBlockGroupFips as GEOIDs are updated with each census)
    param: preallocate_per_record: integer denoting high estimate of number of matching buildings per NFIP record. 
    param: preallocate_cutoff: number of records below which to switch to a safer preallocation method. 
    returns: matched_records: pandas dataframe where each row corresponds to a pair of matching records. 
    returns: unmatched_records: list of indices in left dataframe that failed to match. 
    """

    # Create boolean array that we'll use to keep track of which columns have multiple allowed values
    # in right dataframe
    multiple_value_indicator = [x in multiple_value_cols for x in matching_cols]

    # Create list of records that don't match to a building
    unmatched_records = []
    
    # Records that have missing values in any of the columns used for matching 
    missing_mask = left[matching_cols].isna().any(axis=1)
    unmatched_records += list(left[missing_mask].index.values)
    
    # Drop records with missing values from this round of the matching procedure
    # (Can later attempt to match these using a smaller subset of columns)
    left = left[~missing_mask]
    
    # Preallocate arrays to store matching information
    if len(left) > preallocate_cutoff:
        N_prealloc = len(left)*preallocate_per_record
    else:
        N_prealloc = len(left)*len(right)
    
    left_match_indices = np.empty(N_prealloc,dtype=left.index.to_numpy().dtype)
    right_match_indices = np.empty(N_prealloc,dtype=right.index.to_numpy().dtype)
    match_key_arr = np.empty(N_prealloc,dtype='<U350')
    
    cumulative_num_matches = 0
    
    for i in range(len(left)):
        
        record = left.iloc[i]
        
        # Initialize logical array used for filtering
        mask = pd.Series(True,index=right.index)
        
        for col,mv in zip(matching_cols,multiple_value_indicator):
        
            if mv:
                # If multiple values allowed, check if left value shows up in right record's string of comma-separated allowed values.
                # A more generalizable way to do this would be to first convert the string literal to an actual list in python; 
                # however, doing so results in runtimes that are 4x longer. The current method is relatively fast, but would fail
                # in situations where you might have identifiers that can be substrings of one another 
                # (e.g., '123' would match to '123', '0123', and '1234'). Because we are mainly interested in identifiers
                # such as census GEOIDS (which are all the same length and unique), this should not be a problem for 
                # this project. This assumption may not be valid for other types of identifiers.  
                mask &= (right[f'{col}_values'].str.contains(record[col]))
            else:
                # If exact match required, check for equality between left value and right value
                mask &= (right[col] == record[col])

        # Get number of potential matches in right dataframe
        num_matches = np.sum(mask)

        if num_matches == 0:

            # If no matching records in right dataframe, add to list of unmatched records
            # (can try again in next round using a subset of columns)
            unmatched_records += [record.name]
        
        else:

            # Keep track of potential matches between right and left dataframe
            left_match_indices[cumulative_num_matches:(cumulative_num_matches+num_matches)] = record.name
            right_match_indices[cumulative_num_matches:(cumulative_num_matches+num_matches)] = right[mask].index.values

            # Record attributes used for matching
            match_key_string = '&'.join([f'({col}=={record[col]})' for col in matching_cols])
            match_key_arr[cumulative_num_matches:(cumulative_num_matches+num_matches)] = match_key_string

            # Increment counter
            cumulative_num_matches += num_matches

    # Drop preallocated elements that we didn't end up using
    left_match_indices = left_match_indices[:cumulative_num_matches]
    right_match_indices = right_match_indices[:cumulative_num_matches]
    match_key_arr = match_key_arr[:cumulative_num_matches]

    # Save as dataframe
    matched_records = pd.DataFrame({'left_index':left_match_indices,'right_index':right_match_indices,'match_key':match_key_arr})
    
    return(matched_records,unmatched_records)

### *** INITIAL SETUP *** ###

pwd = os.getcwd()

# Specify state of interest
state_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
state_list_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/CONUS_state_list.txt'
state_list = np.loadtxt(state_list_path,dtype=str)
state = state_list[state_idx]

# Set up folder for output
# Create output folder if it doesn't already exist 
outfolder = os.path.join(pwd,'potential_matches',state,'policies')
if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)

### *** LOAD DATA SOURCES *** ###

# Building points
buildings_path = f'/proj/characklab/projects/kieranf/flood_damage_index/data/NSI/{state}/{state}_structure_info.parquet'
buildings = pd.read_parquet(buildings_path)

# OpenFEMA NFIP policy data
policies_path = '/proj/characklab/projects/kieranf/OpenFEMA/FimaNfipPolicies.parquet'
usecols = ['id','propertyState','latitude','longitude','censusBlockGroupFips','reportedZipCode','ratedFloodZone','occupancyType']
filters = [('propertyState','=',state)]
policies = pd.read_parquet(policies_path,columns=usecols,filters=filters)

### *** FORMAT AND CLEAN DATA *** ###

# Drop records with missing latitude and longitude
# (will keep track of their ids in case we want to do something with them later)
missing_coordinate_mask = policies[['latitude','longitude']].isna().any(axis=1)
policies_missing_coordinate_ids = policies[missing_coordinate_mask]['id'].to_list()
policies = policies[~missing_coordinate_mask]
bad_geocode_records = pd.DataFrame({'openfema_policy_id':policies_missing_coordinate_ids,'nsi_fd_id':pd.NA,'match_key':pd.NA})
outname = os.path.join(outfolder,f'{state}_policy_missing_latlon.parquet')
bad_geocode_records.to_parquet(outname)

# Create columns that we'll use when assigning policies to buildings
# (should already be present in building points dataset)
policies['match_latitude'] = policies['latitude'].apply(lambda x: f'{x:.1f}')
policies['match_longitude'] = policies['longitude'].apply(lambda x: f'{x:.1f}')
policies['match_censusBlockGroupFips'] = policies['censusBlockGroupFips'].copy()
policies['match_sfhaIndicator'] = policies['ratedFloodZone'].apply(is_sfha_zone)
policies['match_coastalFloodZoneIndicator'] = policies['ratedFloodZone'].apply(is_coastal_flood_zone)
policies['match_reportedZipCode'] = policies['reportedZipCode'].copy()
policies['match_simplifiedOccupancyType'] = policies['occupancyType'].apply(simplify_openfema_occupancy_types)

# Divide data into "chunks" based on rounded lat/lon that we can use to break the 
# computationally demanding task of identifying potential matches between records
# and buildings into smaller pieces. This also has the added benefit of allowing us 
# to resume processing at the previous chunk if the job times out. 

policies['chunk'] = '(' + policies['match_latitude']  + ',' + policies['match_longitude'] + ')'
buildings['chunk'] = '(' + buildings['match_latitude']  + ',' + buildings['match_longitude'] + ')'

# Get list of all unique chunks
chunks = np.sort(policies['chunk'].unique())

# Determine which ones we've already processed in a previous job
# (Filename will include the chunk identifier)
completed_chunks_filepath = os.path.join(outfolder,'completed_chunks.txt')
if os.path.exists(completed_chunks_filepath):
    completed_chunks = np.sort(np.loadtxt(completed_chunks_filepath,dtype=str))
else:
    completed_chunks = []


# Get list of chunks that haven't been processed yet
chunks_to_process = [c for c in chunks if c not in completed_chunks]

num_chunks = len(chunks)
num_completed_chunks = len(completed_chunks)

### *** MATCH OPENFEMA NFIP RECORDS TO BUILDINGS *** ###

for i,chunk in enumerate(chunks_to_process):

    print(f'\n\n*** {state} Policies Chunk {num_completed_chunks + i + 1} / {num_chunks}: {chunk} ***\n')

    policy_chunk_mask = (policies['chunk']==chunk)
    building_chunk_mask = (buildings['chunk']==chunk)

    num_records = np.sum(policy_chunk_mask)
    num_buildings = np.sum(building_chunk_mask)

    # Pull out NFIP records and building points that fall inside chunk grid cell
    # Also set the index of these dataframes to be uniquely identifying
    left = policies[policy_chunk_mask].copy().set_index('id')
    right = buildings[building_chunk_mask].copy().set_index('fd_id')

    print(f'Number of NFIP records: {num_records}',flush=True)
    print(f'Number of building points: {num_buildings}',flush=True)

    if num_buildings > 0:

        # Specify which columns to use for matching.
        # List these in order of priority, with most important coming first.
        # At the end of each round, if any unmatched records remain, we'll eliminate the 
        # Least important column from this list, and attempt to get a less precise match. 
        
        matching_cols = ['match_latitude',
                         'match_longitude',
                         'match_censusBlockGroupFips',
                         'match_sfhaIndicator',
                         'match_coastalFloodZoneIndicator',
                         'match_reportedZipCode',
                         'match_simplifiedOccupancyType']

        matched_records_list = []
        unmatched_records = list(left.index.values)
        keepgoing = True

        while keepgoing:

            left = left[left.index.isin(unmatched_records)]
            matched_records,unmatched_records = find_matching_records(left,right,matching_cols,multiple_value_cols=['match_censusBlockGroupFips'])
            matched_records_list.append(matched_records.copy())

            if (len(unmatched_records) == 0) or (len(matching_cols) <= 1):
                # Stop if all records are matched or if the number of columns
                # used for matching has been whittled down to nothing
                keepgoing = False
            else:
                # Otherwise, attempt to perform a less precise match for 
                # remaining unmatched records, removing the least important
                # data field used for matching that still remains
                matching_cols.pop(-1)

        chunk_matched_records = pd.concat(matched_records_list)
        chunk_unmatched_records = pd.DataFrame({'left_index':unmatched_records,'right_index':pd.NA,'match_key':pd.NA})
        chunk_info = pd.concat([chunk_matched_records,chunk_unmatched_records]).reset_index(drop=True)
    
        # Print statistics describing output of matching procedure
        num_records_matched = len(chunk_matched_records['left_index'].unique())
        num_records_unmatched = len(chunk_unmatched_records)
        Q1,Q2,Q3 = chunk_matched_records.groupby('left_index')['right_index'].count().quantile([0.25,0.5,0.75])
    
        print(f'Number of NFIP records matched to at least one building: {num_records_matched}',flush=True)
        print(f'Number of NFIP records with no matching buildings: {len(unmatched_records)}',flush=True)
        print(f'Median (IQR) number of matching buildings per NFIP record: {int(Q2)} ({int(Q1)}–{int(Q3)})')
                
    else:
        chunk_info = pd.DataFrame({'left_index':left.index.values,'right_index':pd.NA,'match_key':pd.NA})
        print(f'No buildings in {state} fall within {chunk}, NFIP geocode is likely incorrect',flush=True)

    # Break match info into separate policy and building dataframes that can be joined via the match_key
    # (this is a more efficient way to store the information) 
    chunk_info.rename(columns = {'left_index':'openfema_policy_id','right_index':'nsi_fd_id'},inplace=True)
    chunk_policy_info = chunk_info.groupby('openfema_policy_id').agg({'match_key':['first','count']})
    chunk_policy_info.columns = ['match_key','num_matches']
    chunk_policy_info = chunk_policy_info.sort_values(by='match_key').reset_index()
    chunk_building_info = chunk_info[['match_key','nsi_fd_id']].dropna().drop_duplicates().sort_values(by='match_key').reset_index(drop=True)

    # Save results to file
    outname = os.path.join(outfolder,f'{state}_policy_matching_policy_info_{chunk}.parquet')
    chunk_policy_info.to_parquet(outname)

    # Save results to file
    outname = os.path.join(outfolder,f'{state}_policy_matching_building_info_{chunk}.parquet')
    chunk_building_info.to_parquet(outname)

    with open(completed_chunks_filepath, 'a') as f:
        f.write(f'{chunk}\n')

### *** CONCATENATE RESULTS *** ###

filepaths = [os.path.join(outfolder,x) for x in np.sort(os.listdir(outfolder)) if x.endswith('.parquet')]
policy_info_filepaths = [x for x in filepaths if 'policy_info' in x]
building_info_filepaths = [x for x in filepaths if 'building_info' in x]

state_policy_info = pd.concat([pd.read_parquet(f) for f in policy_info_filepaths]).reset_index(drop=True)
outname = os.path.join(pwd,'potential_matches',state,f'{state}_policy_matching_policy_info.parquet')
state_policy_info.to_parquet(outname)

state_building_info = pd.concat([pd.read_parquet(f) for f in building_info_filepaths]).reset_index(drop=True)
outname = os.path.join(pwd,'potential_matches',state,f'{state}_policy_matching_building_info.parquet')
state_building_info.to_parquet(outname)