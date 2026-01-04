import numpy as np
import pandas as pd
import os

### *** INITIAL SETUP *** ###

pwd = os.getcwd()

outfolder = os.path.join(pwd,'presence_absence_data')
if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)

### *** LOAD DATA *** ###

## Historical event catalog
event_catalog_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/event_delineation/historical_TC_event_info.csv'
event_catalog = pd.read_csv(event_catalog_path)
event_catalog['START_DATE'] = pd.to_datetime(event_catalog['START_DATE'])
event_catalog['END_DATE'] = pd.to_datetime(event_catalog['END_DATE'])

## Counties included in study area
included_GEOIDs_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/event_delineation/included_county_GEOIDs.txt'
included_GEOIDs = np.loadtxt(included_GEOIDs_path,dtype=str)

## OpenFEMA NFIP records

# Filter by study area counties
filters = [('countyCode','in',included_GEOIDs)]

# Claims
claims_path = '/proj/characklab/projects/kieranf/OpenFEMA/FimaNfipClaims.parquet'
usecols = ['id','countyCode','dateOfLoss']
claims = pd.read_parquet(claims_path,columns=usecols,filters=filters).rename(columns={'id':'openfema_claim_id'})

# Policies
policies_path = '/proj/characklab/projects/kieranf/OpenFEMA/FimaNfipPolicies.parquet'
usecols = ['id','countyCode','policyEffectiveDate','policyTerminationDate']
policies = pd.read_parquet(policies_path,columns=usecols,filters=filters).rename(columns={'id':'openfema_policy_id'})

## Potential building candidates for NFIP records
claim_match_info_path = os.path.join(pwd,'potential_matches/CONUS_nfip_matching_claim_info.parquet')
claim_match_info = pd.read_parquet(claim_match_info_path)
policy_match_info_path = os.path.join(pwd,'potential_matches/CONUS_nfip_matching_policy_info.parquet')
policy_match_info = pd.read_parquet(policy_match_info_path)
building_lookup_path = os.path.join(pwd,'potential_matches/CONUS_nfip_matching_building_lookup.parquet')
building_lookup = pd.read_parquet(building_lookup_path)

### *** DETERMINE NUMBER OF PRESENCE-ABESNCE POINTS BY EVENT *** ###

# Drop claim and policy records that did not match to any builings
# (extremely tiny number, likely due to errors in OpenFEMA's geolocation process) 
valid_claim_ids = claim_match_info[claim_match_info['num_matches']>=1]['openfema_claim_id'].to_numpy()
valid_policy_ids = policy_match_info[policy_match_info['num_matches']>=1]['openfema_policy_id'].to_numpy()
claims = claims[claims['openfema_claim_id'].isin(valid_claim_ids)]
policies = policies[policies['openfema_policy_id'].isin(valid_policy_ids)]

# Attach match_key information to claim and policy records
claims = pd.merge(claims,claim_match_info[['openfema_claim_id','match_key','num_matches']],on='openfema_claim_id',how='left')
policies = pd.merge(policies,policy_match_info[['openfema_policy_id','match_key','num_matches']],on='openfema_policy_id',how='left')

## First deal with post-2009 events
# (These are simpler since we have reliable policy data in OpenFEMA)
# (Can add earlier events later) 

post_2009_mask = (event_catalog['SEASON'] > 2009)
post_2009_event_numbers = event_catalog[post_2009_mask]['EVENT_NUMBER'].to_list()

group_counts_list = []

for event_number in post_2009_event_numbers:
    event_info = event_catalog[event_catalog['EVENT_NUMBER']==event_number].iloc[0]
    start_date = event_info['START_DATE']
    end_date = event_info['END_DATE']

    # Get claims that occur between start/end dates of event
    claim_filter = (claims['dateOfLoss'] >= start_date)&(claims['dateOfLoss'] <= end_date)
    event_claims = claims[claim_filter]

    # Get policies that were in force during middle of event
    # (we'll use this timepoint to assess coverage)
    mid_date = start_date + (end_date-start_date)/2
    policy_filter = (policies['policyEffectiveDate'] <= mid_date)&(policies['policyTerminationDate'] >= mid_date)
    event_policies = policies[policy_filter]

    event_claims['count'] = 1
    event_policies['count'] = 1
    
    claim_match_keys = event_claims['match_key'].drop_duplicates()
    policy_match_keys = event_policies['match_key'].drop_duplicates()
    
    unique_match_keys = np.sort(np.unique(np.concatenate((claim_match_keys,policy_match_keys))))
    
    group_counts = pd.DataFrame(index=unique_match_keys)
    group_counts.index.name = 'match_key'
    group_counts['EVENT_NUMBER'] = event_number
    
    group_counts['num_policies'] = event_policies.groupby('match_key')['count'].sum()
    group_counts['num_claims'] = event_claims.groupby('match_key')['count'].sum()
    
    group_counts.fillna(0,inplace=True)
    group_counts[['num_policies','num_claims']] = group_counts[['num_policies','num_claims']].astype(int)
    group_counts.reset_index(inplace=True)
    
    policy_building_counts = event_policies[['match_key','num_matches']].drop_duplicates().rename(columns={'num_matches':'num_buildings'})
    claim_building_counts = event_claims[['match_key','num_matches']].drop_duplicates().rename(columns={'num_matches':'num_buildings'})
    building_counts = pd.concat([policy_building_counts,claim_building_counts]).drop_duplicates().reset_index(drop=True)
    
    group_counts = pd.merge(group_counts,building_counts,on='match_key',how='left')
    
    group_counts['num_presence'] = np.minimum(group_counts['num_claims'],group_counts['num_buildings'])
    group_counts['num_absence'] = np.maximum(group_counts['num_policies'] - group_counts['num_claims'],0)
    group_counts['num_absence'] = np.minimum(group_counts['num_absence'],group_counts['num_buildings'] - group_counts['num_presence'])
    
    num_policies = group_counts['num_policies'].sum()
    num_claims = group_counts['num_claims'].sum()
    num_presence = group_counts['num_presence'].sum()
    num_absence = group_counts['num_absence'].sum()
    
    pa_point_ratio = (num_presence + num_absence)/num_policies
    
    print(f'\n\n*** EVENT #{event_number}: {event_info['ASSOCIATED_NAMES']} ({event_info['SEASON']}) ***\n',flush=True)
    print(f'Number of NFIP polices: {num_policies}',flush=True)
    print(f'Number of NFIP claims: {num_claims}',flush=True)
    print(f'Number of presence points: {num_presence}',flush=True)
    print(f'Number of absence points: {num_absence}',flush=True)
    print(f'Ratio of presence-absence points to policies: {100*pa_point_ratio:.2f}%',flush=True)

    group_counts_list.append(group_counts)

combined_group_counts = pd.concat(group_counts_list).reset_index(drop=True)

### *** SAVE RESULTS *** ###

outname = os.path.join(outfolder,'presence_absence_summary.parquet')
combined_group_counts.to_parquet(outname)