import numpy as np
import pandas as pd
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

def determine_match_keys(left_record,right,matching_cols,multiple_value_cols=[]):

    """
    This function determines the combination of attributes that can be used to match
    a given OpenFEMA record to building points. 

    param: left_record: row of pandas dataframe of NFIP claim or policy records. Index should be unique. 
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
    returns: match_key: string representing unique combination of attributes used to match left_record to entries in right dataframe. 
    returns: match_type: string describing columns used to create match. 
    returns: match_precision: number of attributes used to match left_record. 
    returns: num_matches: number of records in right dataframe with the same attributes as left_record. 
    """

    # If a record has missing attributes, remove those from columns used for matching
    matching_cols = [x for x in matching_cols if not pd.isna(left_record[x])]

    multiple_value_indicator = [x in multiple_value_cols for x in matching_cols]
    match_key = pd.NA
    match_type = pd.NA
    num_matches = 0

    # Initialize logical array used for filtering
    match_filter = np.full((len(right),len(matching_cols)),False)

    # Populate each column of logical array based on different critiera used for matching
    for i,col in enumerate(matching_cols):
        
        if multiple_value_indicator[i]:
            
            # If multiple values allowed, check if left value shows up in right record's string of comma-separated allowed values.
            # A more generalizable way to do this would be to first convert the string literal to an actual list in python; 
            # however, doing so results in runtimes that are 4x longer. The current method is relatively fast, but would fail
            # in situations where you might have identifiers that can be substrings of one another 
            # (e.g., '123' would match to '123', '0123', and '1234'). Because we are mainly interested in identifiers
            # such as census GEOIDS (which are all the same length and unique), this should not be a problem for 
            # this project. This assumption may not be valid for other types of identifiers. 
            
            match_filter[:,i] = (right[f'{col}_values'].str.contains(left_record[col])).fillna(False).to_numpy()
            
        else:
            # If exact match required, check for equality between left value and right value
            match_filter[:,i] = (right[col] == left_record[col]).fillna(False).to_numpy()

    match_precision = len(matching_cols)
    keepgoing = True

    while keepgoing:

        # Check whether there's any records in right dataframe that satisfy all criteria
        num_matches = match_filter[:,:match_precision].all(axis=1).sum()

        if num_matches > 0:
            # If so, stop and record criteria used to identify matches
            match_key = '&'.join([f'({col}=={left_record[col]})' for col in matching_cols[:match_precision]])
            match_type = '+'.join(matching_cols[:match_precision])
            keepgoing = False
        elif (match_precision > 0):
            # If not, eliminate the least important criteria
            match_precision -= 1
        else:
            # Stop if there's no more criteria left to eliminate
            keepgoing = False

    return(match_key,match_type,match_precision,num_matches)

def create_lookup_table(match_record,right,multiple_value_cols=[]):
    """
    This function returns the records in the right dataframe associated with a given match_key. 
    
    param: match_record: row of pandas dataframe describing characteristics of each match_key. 
    param: pandas dataframe of building points. Index of each row should be unique. 
    param: multiple_value_cols: list of columns in the right dataframe that are allowed to take on
                                multiple values (e.g., a building can have multiple 
                                censusBlockGroupFips as GEOIDs are updated with each census)
    returns: dataframe listing indices of records in right dataframe associated with a given match_key. 
    """

    # Get attributes used for matching
    matching_cols = match_record['match_type'].split('+')
    multiple_value_indicator = [x in multiple_value_cols for x in matching_cols]

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
            mask &= (right[f'{col}_values'].str.contains(match_record[col]))
        else:
            # If exact match required, check for equality
            mask &= (right[col] == match_record[col])

    right_indices = right[mask].index.values

    return(pd.DataFrame({'match_key':match_record['match_key'],'right_index':right_indices}))

def identify_potential_matches(left,right,matching_cols,multiple_value_cols=[]):
    """
    This function identifies potential matches between OpenFEMA records and building points. 

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

    returns: left_match_info: dataframe describing match keys associated with each record in left dataframe, 
                            plus other information about the match. 
    returns: right_match_info: dataframe that describes building ids associated with each match key. 
    """

    # Enumerate unique combinations of matching attributes observed in data
    # We'll use this number to group together rows that have the same attributes
    left.sort_values(by=matching_cols,inplace=True)
    left['group_key'] = (~left[matching_cols].duplicated()).astype(int).cumsum()
    
    # For each of these groups, we only need to search for matching buildings once
    # (much more efficient than doing a search for each record)
    left_groups = left.drop_duplicates(subset=matching_cols).set_index('group_key')
    
    # Identify potential building matches for each attribute combination 
    left_groups[['match_key','match_type','match_precision','num_matches']] = left_groups.apply(determine_match_keys,args=(right,matching_cols),multiple_value_cols=multiple_value_cols,axis=1,result_type='expand')
    
    # Propagate match info back to each record based on group number
    left_match_info = left_groups.loc[left['group_key'],['match_key','match_type','match_precision','num_matches']]
    left_match_info.index = left.index
    left_match_info.index.name = 'left_index'
    left_match_info.reset_index(inplace=True)

    # Get list of unique match_keys and associated attributes
    # (might be smaller than number of left groups due to buildings matching on subset of columns)
    match_key_info = left_groups[left_groups['num_matches'] > 0]
    match_key_info = match_key_info[['match_key','match_type','match_precision','num_matches']+matching_cols].groupby('match_key').first().reset_index()

    # Create table that will allow us to quickly look up the indices of records 
    # in the right dataframe that are associated with a given match_key
    right_match_info = pd.concat([create_lookup_table(match_record,right,multiple_value_cols=multiple_value_cols) for i,match_record in match_key_info.iterrows()]).reset_index(drop=True)
    
    return(left_match_info,right_match_info)

### *** INITIAL SETUP *** ###

pwd = os.getcwd()

# Specify state of interest
state_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
state_list_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/CONUS_state_list.txt'
state_list = np.loadtxt(state_list_path,dtype=str)
state = state_list[state_idx]

# Set up folder for output
# Create output folder if it doesn't already exist 
outfolder = os.path.join(pwd,'potential_matches',state,'claims')
if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)

### *** LOAD DATA SOURCES *** ###

# Building points
buildings_path = f'/proj/characklab/projects/kieranf/flood_damage_index/data/NSI/{state}/{state}_structure_info.parquet'
buildings = pd.read_parquet(buildings_path)

# OpenFEMA NFIP claims data
claims_path = '/proj/characklab/projects/kieranf/OpenFEMA/FimaNfipClaims.parquet'
usecols = ['id','state','latitude','longitude','censusBlockGroupFips','reportedZipCode','ratedFloodZone','occupancyType']
filters = [('state','=',state)]
claims = pd.read_parquet(claims_path,columns=usecols,filters=filters)

### *** FORMAT AND CLEAN DATA *** ###

# Drop records with missing latitude and longitude
# (will keep track of their ids in case we want to do something with them later)
missing_coordinate_mask = claims[['latitude','longitude']].isna().any(axis=1)
claims_missing_coordinate_ids = claims[missing_coordinate_mask]['id'].to_list()
claims = claims[~missing_coordinate_mask]
bad_geocode_records = pd.DataFrame({'openfema_claim_id':claims_missing_coordinate_ids,'nsi_fd_id':pd.NA,'match_key':pd.NA})
outname = os.path.join(outfolder,f'{state}_claim_missing_latlon.parquet')
bad_geocode_records.to_parquet(outname)

# Create columns that we'll use when assigning claims to buildings
# (should already be present in building points dataset)
claims['match_latitude'] = claims['latitude'].apply(lambda x: f'{x:.1f}')
claims['match_longitude'] = claims['longitude'].apply(lambda x: f'{x:.1f}')
claims['match_censusBlockGroupFips'] = claims['censusBlockGroupFips'].copy()
claims['match_sfhaIndicator'] = claims['ratedFloodZone'].apply(is_sfha_zone)
claims['match_coastalFloodZoneIndicator'] = claims['ratedFloodZone'].apply(is_coastal_flood_zone)
claims['match_reportedZipCode'] = claims['reportedZipCode'].copy()
claims['match_simplifiedOccupancyType'] = claims['occupancyType'].apply(simplify_openfema_occupancy_types)

# Divide data into "chunks" based on rounded lat/lon that we can use to break the 
# computationally demanding task of identifying potential matches between records
# and buildings into smaller pieces. This also has the added benefit of allowing us 
# to resume processing at the previous chunk if the job times out. 

claims['chunk'] = '(' + claims['match_latitude']  + ',' + claims['match_longitude'] + ')'
buildings['chunk'] = '(' + buildings['match_latitude']  + ',' + buildings['match_longitude'] + ')'

# Get list of all unique chunks
chunks = np.sort(claims['chunk'].unique())

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

    print(f'\n\n*** {state} Claims Chunk {num_completed_chunks + i + 1} / {num_chunks}: {chunk} ***\n')

    claim_chunk_mask = (claims['chunk']==chunk)
    building_chunk_mask = (buildings['chunk']==chunk)

    num_records = np.sum(claim_chunk_mask)
    num_buildings = np.sum(building_chunk_mask)

    # Pull out NFIP records and building points that fall inside chunk grid cell
    # Also set the index of these dataframes to be uniquely identifying
    left = claims[claim_chunk_mask].copy().set_index('id')
    right = buildings[building_chunk_mask].copy().set_index('fd_id')

    print(f'Number of NFIP records: {num_records}',flush=True)
    print(f'Number of building points: {num_buildings}',flush=True)

    if num_buildings > 0:

        # Attempt to match OpenFEMA records to buildings
        
        matching_cols = ['match_latitude',
                 'match_longitude',
                 'match_censusBlockGroupFips',
                 'match_sfhaIndicator',
                 'match_coastalFloodZoneIndicator',
                 'match_reportedZipCode',
                 'match_simplifiedOccupancyType']

        multiple_value_cols = ['match_censusBlockGroupFips']
        
        left_match_info,right_match_info = identify_potential_matches(left,right,matching_cols,multiple_value_cols=multiple_value_cols)

        # Print statistics describing output of matching procedure
        matched_mask = (left_match_info['num_matches']>0)
        num_records_matched = np.sum(matched_mask)
        num_records_unmatched = np.sum(~matched_mask)
        Q1,Q2,Q3 = left_match_info[matched_mask]['num_matches'].quantile([0.25,0.5,0.75])
    
        print(f'Number of NFIP records matched to at least one building: {num_records_matched}',flush=True)
        print(f'Number of NFIP records with no matching buildings: {num_records_unmatched}',flush=True)
        print(f'Median (IQR) number of matching buildings per NFIP record: {int(Q2)} ({int(Q1)}–{int(Q3)})')
                
    else:
        left_match_info = pd.DataFrame({'left_index':left.index.values,'match_key':pd.NA,'match_type':pd.NA,'match_precision':0,'num_matches':0})
        right_match_info = pd.DataFrame({'match_key':[pd.NA],'right_index':[pd.NA]})
        print(f'No buildings in {state} fall within {chunk}, NFIP geocode is likely incorrect',flush=True)

    # Rename record unique identifiers to be more interpretable
    left_match_info.rename(columns={'left_index':'openfema_claim_id'},inplace=True)
    right_match_info.rename(columns={'right_index':'nsi_fd_id'},inplace=True)
    
    # Save results to file
    outname = os.path.join(outfolder,f'{state}_claim_matching_claim_info_{chunk}.parquet')
    left_match_info.to_parquet(outname)

    # Save results to file
    if num_buildings > 0:
        outname = os.path.join(outfolder,f'{state}_claim_matching_building_info_{chunk}.parquet')
        right_match_info.to_parquet(outname)

    with open(completed_chunks_filepath, 'a') as f:
        f.write(f'{chunk}\n')

## *** CONCATENATE RESULTS *** ###

filepaths = [os.path.join(outfolder,x) for x in np.sort(os.listdir(outfolder)) if x.endswith('.parquet')]
claim_info_filepaths = [x for x in filepaths if 'claim_info' in x]
building_info_filepaths = [x for x in filepaths if 'building_info' in x]

state_claim_info = pd.concat([pd.read_parquet(f) for f in claim_info_filepaths]).reset_index(drop=True)
outname = os.path.join(pwd,'potential_matches',state,f'{state}_claim_matching_claim_info.parquet')
state_claim_info.to_parquet(outname)

state_building_info = pd.concat([pd.read_parquet(f) for f in building_info_filepaths]).reset_index(drop=True)
outname = os.path.join(pwd,'potential_matches',state,f'{state}_claim_matching_building_info.parquet')
state_building_info.to_parquet(outname)

