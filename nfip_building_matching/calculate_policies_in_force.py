import numpy as np
import pandas as pd
import os

### *** HELPER FUNCTIONS *** ###

def is_sfha_zone(x):
    """
    This function determines if the flood zone code (e.g., AE) listed in OpenFEMA corresponds to the SFHA.
    Returns 1 for SFHA zones, and zero otherwise. 
    """
    if not pd.isna(x):
        return int(x.upper().startswith('A') or x.upper().startswith('V'))
    else:
        return 0

def policies_in_force_over_time(df):

    t = pd.date_range('2009-01-01','2025-01-01',freq='D').astype('date32[day][pyarrow]')
    timeseries_df = pd.DataFrame(data={'date':t})

    inflow = df[['policyEffectiveDate','policyCount']].groupby('policyEffectiveDate').sum().reset_index().rename(columns={'policyCount':'inflow','policyEffectiveDate':'date'})
    outflow = df[['policyTerminationDate','policyCount']].groupby('policyTerminationDate').sum().reset_index().rename(columns={'policyCount':'outflow','policyTerminationDate':'date'})

    timeseries_df = pd.merge(timeseries_df,inflow,on='date',how='left')
    timeseries_df = pd.merge(timeseries_df,outflow,on='date',how='left')
    timeseries_df.fillna(0,inplace=True)

    timeseries_df['netflow'] = timeseries_df['inflow'] - timeseries_df['outflow']
    timeseries_df['policies_in_force'] = timeseries_df['netflow'].cumsum()

    return(timeseries_df)

def claims_over_time(df):

    df['claim_count'] = 1

    t = pd.date_range('1990-01-01','2025-01-01',freq='D').astype('date32[day][pyarrow]')
    timeseries_df = pd.DataFrame(data={'date':t})
    
    daily_claims = df[['dateOfLoss','claim_count']].groupby('dateOfLoss').sum()
    daily_claims.index.name = 'date'
    daily_claims.reset_index(inplace=True)
    
    timeseries_df = pd.merge(timeseries_df,daily_claims,on='date',how='left').fillna(0)

    return(timeseries_df)

### *** INITIAL SETUP *** ###

pwd = os.getcwd()

outfolder = os.path.join(pwd,'nfip_timeseries')
if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)

### *** LOAD DATA SOURCES *** ###

## OpenFEMA data

claims_path = '/proj/characklab/projects/kieranf/OpenFEMA/FimaNfipClaims.parquet'
usecols = ['id','state','countyCode','ratedFloodZone','dateOfLoss']
claims = pd.read_parquet(claims_path,columns=usecols)
claims['dateOfLoss'] = pd.to_datetime(claims['dateOfLoss']).astype('date32[day][pyarrow]')
claims['sfhaIndicator'] = claims['ratedFloodZone'].apply(is_sfha_zone)

policies_path = '/proj/characklab/projects/kieranf/OpenFEMA/FimaNfipPolicies.parquet'
usecols = ['id','propertyState','countyCode','ratedFloodZone','policyEffectiveDate','policyTerminationDate','policyCount']
policies = pd.read_parquet(policies_path,columns=usecols)
policies['policyEffectiveDate'] = pd.to_datetime(policies['policyEffectiveDate']).astype('date32[day][pyarrow]')
policies['policyTerminationDate'] = pd.to_datetime(policies['policyTerminationDate']).astype('date32[day][pyarrow]')
policies['sfhaIndicator'] = policies['ratedFloodZone'].apply(is_sfha_zone)

# Drop those with missing data in specified columns
# (should be tiny number)
claims.dropna(inplace=True)
policies.dropna(inplace=True)

### *** GET DAILY COUNTS *** ###

# Get daily number of claims and policies in force stratified by county and SFHA
policy_timeseries = policies.groupby(['countyCode','sfhaIndicator']).apply(policies_in_force_over_time).reset_index().drop(columns='level_2')
claim_timeseries = claims.groupby(['countyCode','sfhaIndicator']).apply(claims_over_time).reset_index().drop(columns='level_2')

# Save results
outname = os.path.join(outfolder,'daily_nfip_policies_in_force_by_county_SFHA.parquet')
policy_timeseries.to_csv(outname)

outname = os.path.join(outfolder,'daily_nfip_claims_by_county_SFHA.parquet')
claim_timeseries.to_csv(outname)