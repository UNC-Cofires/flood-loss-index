import numpy as np
import pandas as pd
import geopandas as gpd
import pickle
import os

### *** HELPER FUNCTIONS *** ### 

def join_points_to_polygons(points,polygons,max_dist=100):
    
    """
    Helper function to spatially join points to polygons. 
    Starts by using within predicate to speed things up, then does sjoin_nearest for 
    any remaining unmatched points that fall within distance threshold of polygon. 
    """
    
    df1 = points.sjoin(polygons,how='left',predicate='within')
    matched_mask = ~df1['index_right'].isna()
    df1 = df1.drop(columns=['index_right'])
    df1 = df1[matched_mask]
    
    df2 = points[~matched_mask].sjoin_nearest(polygons,how='left',max_distance=max_dist).drop(columns=['index_right'])
    df3 = pd.concat([df1,df2])
    return(df3)

def join_points_to_overlapping_polygons(left,right,id_col):

    """
    Helper function to spatially join points to overlapping polygons. Overlaps commonly occur 
    with national flood hazard layer (NFHL) data. These are mostly benign
    (e.g., a map panel that straddles two counties and is included twice). 
    To maintain the correct number of rows, attributes from overlapping polygons will be aggregated into lists

    param: left: points gdf
    param: right: polygon gdf (may contain overlaps)
    param: id_col: column in points gdf that uniquely identifies each row. 
    """

    df = left[[id_col,'geometry']].copy()
    df = df.sjoin(right,how='left',predicate='within').drop(columns=['index_right'])
    df = df.groupby(id_col).agg(list)
    df.drop(columns='geometry',inplace=True)

    return(pd.merge(left,df,on=id_col,how='left'))

def join_polygons_to_overlapping_polygons(left,right,id_col):

    """
    Helper function to spatially join polygons to overlapping polygons. Overlaps commonly occur 
    with national flood hazard layer (NFHL) data. These are mostly benign
    (e.g., a map panel that straddles two counties and is included twice). 
    To maintain the correct number of rows, attributes from overlapping polygons will be aggregated into lists

    This is also helpful for catching when you have a building whose footprint is partially
    inside the SFHA. 

    param: left: polygon gdf (no overlaps)
    param: right: polygon gdf (may contain overlaps)
    param: id_col: column in polygons gdf that uniquely identifies each row. 
    """

    df = left[[id_col,'polygon_geometry']].copy()
    df = df.sjoin(right,how='left',predicate='intersects').drop(columns=['index_right'])
    df = df.groupby(id_col).agg(list)
    df.drop(columns='polygon_geometry',inplace=True)

    return(pd.merge(left,df,on=id_col,how='left'))

def clean_nan_string_list_values(x):
    """
    If a string representation of a list contains only missing values, return pd.NA
    """
    v = x

    if not pd.isna(x):
        if x in ['[nan]','[NA]','[<NA>]','[]']:
            v = pd.NA

    return v

### *** INITIAL SETUP *** ###

pwd = os.getcwd()

state_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
state_list_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/CONUS_state_list.txt'
state_list = np.loadtxt(state_list_path,dtype=str)

state = state_list[state_idx]
print(state,flush=True)

outfolder = os.path.join(pwd,'structure_info',state)
if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)

# Specify CRS
crs = 'EPSG:5070'

# Read in counties
counties_path = '/proj/characklab/projects/kieranf/flood_damage_index/data/geospatial_data/USA_counties/cb_2022_us_county_500k'
counties = gpd.read_file(counties_path).to_crs(crs)
counties = counties[counties['STUSPS']==state]
state_fips = counties['STATEFP'].values[0]

state_area_gdf = counties[['geometry']].dissolve()

buffer_dist = 10000

state_area_gdf['geometry'] = state_area_gdf['geometry'].buffer(buffer_dist)
state_area_mask = state_area_gdf['geometry'].values[0]

# Read in USA structures data
buildings_path = f'/proj/characklab/projects/kieranf/flood_damage_index/data/USA_structures/{state}/{state}_Structures.gdb'
usecols = ['BUILD_ID','LATITUDE','LONGITUDE','OCC_CLS','PROP_ZIP']
buildings = gpd.read_file(buildings_path,layer=f'{state}_Structures',columns=usecols).to_crs(crs)
buildings.rename(columns={'geometry':'polygon_geometry'},inplace=True)
buildings['geometry'] = buildings['polygon_geometry'].centroid
buildings.set_geometry('geometry',inplace=True)
buildings['state'] = state
buildings['x_epsg5070'] = buildings['geometry'].x
buildings['y_epsg5070'] = buildings['geometry'].y

# Read in old county boundaries
# (need to do this since CT revised maps in 2022)
old_counties_path = '/proj/characklab/projects/kieranf/flood_damage_index/data/geospatial_data/USA_counties/gz_2010_us_050_00_500k'
old_counties = gpd.read_file(old_counties_path).to_crs(crs)
old_counties = old_counties[old_counties['STATE']==state_fips]
old_counties['countyfips_2010'] = old_counties['STATE'] + old_counties['COUNTY']
counties['countyfips_2022'] = counties['GEOID'].copy()

# Read in census block groups
blockgroups_2000_path = '/proj/characklab/projects/kieranf/flood_damage_index/data/geospatial_data/census_blockgroups/2000_census_blockgroups'
blockgroups_2010_path = '/proj/characklab/projects/kieranf/flood_damage_index/data/geospatial_data/census_blockgroups/2010_census_blockgroups'
blockgroups_2020_path = '/proj/characklab/projects/kieranf/flood_damage_index/data/geospatial_data/census_blockgroups/2020_census_blockgroups'

blockgroups_2000 = gpd.read_file(blockgroups_2000_path).to_crs(crs).rename(columns={'GEOID':'censusblockgroup_2000'})
blockgroups_2010 = gpd.read_file(blockgroups_2010_path).to_crs(crs).rename(columns={'GEOID':'censusblockgroup_2010'})
blockgroups_2020 = gpd.read_file(blockgroups_2020_path).to_crs(crs).rename(columns={'GEOID':'censusblockgroup_2020'})

blockgroups_2000 = blockgroups_2000[blockgroups_2000.intersects(state_area_mask)]
blockgroups_2010 = blockgroups_2010[blockgroups_2010.intersects(state_area_mask)]
blockgroups_2020 = blockgroups_2020[blockgroups_2020.intersects(state_area_mask)]

# Read in ZCTAs
ZCTA_path = '/proj/characklab/projects/kieranf/flood_damage_index/data/geospatial_data/census_ZCTAs/tl_2020_us_zcta520'
ZCTAs = gpd.read_file(ZCTA_path).to_crs(crs).rename(columns={'GEOID20':'zcta_2020'})[['zcta_2020','geometry']]

# Read in RPUs
RPU_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/raster_processing/CONUS_raster_processing_units'
RPUs = gpd.read_file(RPU_path).to_crs(crs).rename(columns={'DrainageID':'drainage_id','UnitID':'rpu_id'})
RPUs = RPUs[RPUs.intersects(state_area_mask)]

# Read in HUC6 boundaries
HUC_path = '/proj/characklab/projects/kieranf/flood_damage_index/data/watersheds/CONUS_WBD_HU6'
HUCs = gpd.read_file(HUC_path).to_crs(crs)
HUCs = HUCs[['huc6','geometry']]
HUCs = HUCs[HUCs.intersects(state_area_mask)]

# Read in NFHL data
NFHL_info_path = f'/proj/characklab/projects/kieranf/flood_damage_index/data/NFHL/{state}/{state}_FLD_HAZ_AR.parquet'
NFHL_info = pd.read_parquet(NFHL_info_path)

NFHL_geom_path = f'/proj/characklab/projects/kieranf/flood_damage_index/data/NFHL/{state}/{state}_FLD_HAZ_AR_geometry.pickle'
with open(NFHL_geom_path,'rb') as f:
    NFHL_geom = pickle.load(f)
    
NFHL_geom = NFHL_geom.to_crs(crs)

NFHL = NFHL_geom.merge(NFHL_info,how='left',on='FLD_AR_ID')
NFHL = NFHL[['FLD_ZONE','SFHA_TF','geometry']]

### *** DATA JOINING *** ###

# Get starting number of buildings
print(f'Starting number of buildings: {len(buildings)}',flush=True)

# Spatially join RPUs to buildings
buildings = join_points_to_polygons(buildings,RPUs,max_dist=500)

# Spatially join HUC6 basins to buildings
buildings = join_points_to_polygons(buildings,HUCs,max_dist=500)

# Spatially join 2010 counties to buildings
buildings = join_points_to_polygons(buildings,old_counties[['countyfips_2010','geometry']],max_dist=500)

# Spatially join 2022 counties to buildings
buildings = join_points_to_polygons(buildings,counties[['countyfips_2022','geometry']],max_dist=500)

# Spatially join 2000 block groups to buildings
buildings = join_points_to_polygons(buildings,blockgroups_2000,max_dist=500)

# Spatially join 2010 block groups to buildings
buildings = join_points_to_polygons(buildings,blockgroups_2010,max_dist=500)

# Spatially join 2020 block groups to buildings
buildings = join_points_to_polygons(buildings,blockgroups_2020,max_dist=500)

# Spatially join 2020 ZCTAs to buildings
buildings = join_points_to_polygons(buildings,ZCTAs,max_dist=500)

# Spatially join FEMA flood zone to buildings
# (sometimes flood zone polygons overlap, so may return multiple)
# (also, not all areas are mapped, so may return NA)
buildings.set_geometry('polygon_geometry',crs=crs,inplace=True)
buildings = join_polygons_to_overlapping_polygons(buildings,NFHL,'BUILD_ID')

# Get ending number of buildings (should be same as start)
print(f'Ending number of buildings: {len(buildings)}',flush=True)

### *** DATA CLEANING / FORMATTING *** ###

## Create "cleaned" versions of certain columns that we can use to match with OpenFEMA data.
# Those that will be used for matching will have "match_" at the start of the column name.
# Columns with multiple potential values per building due to overlapping polygons will have
# "_values" appended to the end of the column name. 

for col in ['FLD_ZONE','SFHA_TF']:
    buildings.rename(columns={col:f'NFHL2025_{col}_values'},inplace=True)

buildings['match_latitude'] = buildings['LATITUDE'].apply(lambda x: f'{x:.1f}')
buildings['match_longitude'] = buildings['LONGITUDE'].apply(lambda x: f'{x:.1f}')
buildings['match_countyCode_values'] = buildings[['countyfips_2010','countyfips_2022']].apply(lambda x: list(set(x)),axis=1)
buildings['match_censusBlockGroupFips_values'] = buildings[['censusblockgroup_2000','censusblockgroup_2010','censusblockgroup_2020']].apply(lambda x: list(set(x)),axis=1)
buildings['match_reportedZipCode'] = buildings['PROP_ZIP']
buildings['match_sfhaIndicator'] = buildings['NFHL2025_SFHA_TF_values'].apply(lambda x: 1 if 'T' in x else 0)
buildings['match_coastalFloodZoneIndicator'] = buildings['NFHL2025_FLD_ZONE_values'].apply(lambda x: int(np.sum([str(zone).upper().startswith('V') for zone in x]) > 0))
buildings['match_simplifiedOccupancyType'] = buildings['OCC_CLS'].apply(lambda x: 'R' if x=='Residential' else 'NR')

# If zip code is missing, use ZCTA instead
m = buildings['match_reportedZipCode'].isna()
buildings.loc[m,'match_reportedZipCode'] = buildings.loc[m,'zcta_2020']

### *** SAVE RESULTS *** ###

# Convert to pyarrow dtypes
string_cols = buildings.dtypes[buildings.dtypes=='object'].index.values
for col in string_cols:
    buildings[col] = buildings[col].astype('string[pyarrow]')

# Clean NAN values in lists
buildings['match_countyCode_values'] = buildings['match_countyCode_values'].apply(clean_nan_string_list_values)
buildings['match_censusBlockGroupFips_values'] = buildings['match_censusBlockGroupFips_values'].apply(clean_nan_string_list_values)

# Drop geometry column so we can save as parquet (coordinates already saved as numeric)
buildings.drop(columns=['geometry','polygon_geometry'],inplace=True)

# Save as parquet
outname = os.path.join(outfolder,f'{state}_structure_info.parquet')
buildings.to_parquet(outname)

