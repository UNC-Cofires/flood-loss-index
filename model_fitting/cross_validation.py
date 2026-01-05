import numpy as np
import pandas as pd
import dask.dataframe as dd
import scipy.stats as stats
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
import pickle
import time
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

def format_elapsed_time(seconds):
    """
    Helper function to format elapsed time in seconds to HH:MM:SS. 
    """
    seconds = int(np.round(seconds))
    hours = seconds // 3600
    seconds = seconds - hours*3600
    minutes = seconds // 60
    seconds = seconds - minutes*60
    return(f'{hours}h:{minutes:02d}m:{seconds:02d}s')

def safe_logit(p,smallnum=1e-8):
    """
    Logit transform function that avoids issues related to machine precision and +/- infinity 
    param: smallnum: tiny number to avoid float division by zero
    """
    p = np.clip(p,smallnum,1-smallnum)
    x = np.log(p/(1-p))
    return np.array(x)

### *** PROBABILITY CALIBRATION CLASS *** ###

class PlattScalingCalibrator:
    """
    Helper class that uses Platt scaling to calibrate raw probability estimates of ML models
    """
    
    def __init__(self,beta0=0,beta1=1):
        self.beta0 = beta0
        self.beta1 = beta1

    def fit(self,p,y,sample_weight=None,smallnum=1e-9):
        """
        param: p: raw probability estimate of ML model [numpy array]
        param: y: observed outcome in calibration set [numpy array]
        """

        # Apply logit transform to raw probabilities
        x = safe_logit(p)

        if sample_weight is None:
            sample_weight = np.ones(len(x))

        # Fit coefficients
        mod = LogisticRegression()
        mod.fit(x.reshape(len(x),1),y,sample_weight=sample_weight)
        
        self.beta0 = mod.intercept_[0]
        self.beta1 = mod.coef_[0][0]

    def predict(self,raw_prob):
        """
        param: raw_prob: raw probability estimate of ML model [numpy array]
        param: calibrated_prob: adjusted probability estimate transformed via Platt scaling
        """
        
        x = safe_logit(raw_prob)
        calibrated_prob = 1/(1+np.exp(-1*(self.beta0 + self.beta1*x)))
        return calibrated_prob

### *** FLOOD DAMAGE PREDICTION CLASS *** ###

class FloodDamageProbabilityEstimator:
    
    def __init__(self,hyperparams,
                 outcome_variable,
                 features,
                 categorical_features=[],
                 block_id_col='SPATIAL_BLOCK_ID'):
        """
        param: hyperparams: dictionary of hyperparameters to be used by LGBMClassifier [dict]. 
        param: outcome_variable: name of binary variable representing presence or absence of flood damage [string].
        param: features: list of variables used to predict outcome [list of strings].
        param: categorical_features: list of variables that are categorical [list of strings]. When fitting 
                        and applying models to data, these variables should be cast as the pandas "category" 
                        data type. 
        param: block_id_col: name of variable that identifies the spatial and/or temporal block that 
                        each building belongs to [string]. When fitting the model and calibrating 
                        probabilites, we'll do an inner cross-validation loop that utilizes blocked 
                        cross-validation. This helps to overcome issues related to spatial and/or 
                        temporal autocorrelation in the data. For more info, see doi:10.1111/ecog.02881. 
         """

        # Save information on outcome variable and features
        self.outcome_variable = outcome_variable
        self.outcome_prob_variable = f'{outcome_variable}_prob'
        self.features = features
        self.categorical_features = []
        self.block_id_col = block_id_col

        # Save LGBMClassifier hyperparams
        self.hyperparams = hyperparams
        self.model = None
        self.calibrator = None

        # If the user has increased the amount of weight given to positive 
        # training examples, keep track of this since it will affect how we
        # evaluate performance in the inner CV loop. 
        if 'scale_pos_weight' in self.hyperparams.keys():
            self.scale_pos_weight = self.hyperparams['scale_pos_weight']
        else:
            self.scale_pos_weight = 1.0

    def fit_probabilities(self,data,num_calibration_folds=5,t0=None):
        """
        This function fits an LGBMClassifier model to the data while using an inner
        cross-validation loop to calibrate flood damage probability projections. 

        param: data: pandas dataframe containing data used for model fitting. Must contain 
                        columns corresponding to outcome_variable, features, and 
                        block_id_col attributes. 
        param: num_calibration_folds: number of inner cross-validation folds to use when 
                        calibrating predicted probabilities to data. 
        param: t0: time since start of process [float]. Encoded as # seconds since Unix epoch. 
                        Users can pass this as an argument if the function call is part of a 
                        larger workflow. 
        """

        # If the user hasn't already done so, start counting time. 
        if t0 is None:
            t0 = time.time()

        # Set up inner spatial and/or temporal block CV loop that we'll use to calibrate probabilites 
        block_cv_folds = pd.DataFrame(data[self.block_id_col].unique(),columns=[self.block_id_col])
        block_cv_folds['fold'] = np.random.randint(num_calibration_folds,size=len(block_cv_folds))
        data = pd.merge(data,block_cv_folds,on=self.block_id_col,how='left')

        # Initialize column that we'll use to keep track of predicted probabilities
        data[self.outcome_prob_variable] = np.nan

        # Create weights for eval_set that reflect the degree to which the 
        # user has scaled the weight of positive examples. 
        # (only matters if scale_pos_weight was explicitly passed as a hyperparameter) 
        data['eval_sample_weight'] = np.where(data[self.outcome_variable]==1, self.scale_pos_weight, 1.0)

        # Within each loop, keep track of n_estimators. 
        # This is important for preventing overfitting in final step 
        # where we fit to all the data. 
        best_iteration_list = []

        # Inner CV loop 
        for fold in range(num_calibration_folds):

            t1 = time.time()
        
            print(f'\n\n*** CALIBRATION FOLD {fold+1} / {num_calibration_folds} ***\n',flush=True)
            
            fold_mask = (data['fold']==fold)
            
            model = lgb.LGBMClassifier(**self.hyperparams)
            
            callbacks=[lgb.early_stopping(stopping_rounds=50,min_delta=1e-5),
                       lgb.log_evaluation(period=50)]
            
            model.fit(data[~fold_mask][self.features],
                      data[~fold_mask][self.outcome_variable],
                      feature_name=self.features,
                      categorical_feature=self.categorical_features,
                      eval_set=[(data[fold_mask][self.features], data[fold_mask][self.outcome_variable])],
                      eval_sample_weight=[data[fold_mask]['eval_sample_weight']],
                      eval_metric="binary_logloss",
                      callbacks=callbacks)
        
            best_iteration_list.append(model.best_iteration_)
            
            data.loc[fold_mask,self.outcome_prob_variable] = model.predict_proba(data[fold_mask][self.features])[:,1]

            t2 = time.time()
            
            elapsed_time = format_elapsed_time(t2-t1)
            cumulative_elapsed_time = format_elapsed_time(t2-t0)

            print(f'\nTIME ELAPSED: {elapsed_time} last iteration / {cumulative_elapsed_time} cumulative',flush=True)
        
        print(f'\n\n*** CALIBRATING PROBABILITIES ***\n',flush=True)

        t1 = time.time()
        
        # Use platt scaling to correct for systemic over- or underestimation of probabilities. 
        # This step is particularly important if the user has increased the weight given to 
        # rare positive examples when training the LGBMClassifier, as the raw probabilities 
        # will be biased. 
        calibrator = PlattScalingCalibrator()
        calibrator.fit(data[self.outcome_prob_variable],data[self.outcome_variable])
        self.calibrator = calibrator

        print(f'Platt scaling parameters: beta0 = {calibrator.beta0:.3f}, beta1 = {calibrator.beta1:.3f}',flush=True)

        t2 = time.time()

        elapsed_time = format_elapsed_time(t2-t1)
        cumulative_elapsed_time = format_elapsed_time(t2-t0)

        print(f'\nTIME ELAPSED: {elapsed_time} last iteration / {cumulative_elapsed_time} cumulative',flush=True)
        
        print(f'\n\n*** POST-CALIBRATION MODEL FITTING ***\n',flush=True)

        t1 = time.time()
        
        # Now fit to all the data, and use this to create calibrated probabilities
        n_estimators = int(np.median(best_iteration_list))
        model = lgb.LGBMClassifier(**self.hyperparams)
        model.n_estimators = n_estimators
        
        model.fit(data[self.features],
                  data[self.outcome_variable],
                  feature_name=self.features,
                  categorical_feature=self.categorical_features)

        self.model = model

        t2 = time.time()

        elapsed_time = format_elapsed_time(t2-t1)
        cumulative_elapsed_time = format_elapsed_time(t2-t0)

        print(f'\nTIME ELAPSED: {elapsed_time} last iteration / {cumulative_elapsed_time} cumulative',flush=True)

        return(None)

    def predict_probabilities(self,data):
        """
        param: data: data on which to to predict probabilities [pandas dataframe]. Must follow same 
                        format as data used to fit model. 
        returns: calibrated_prob: predicted probability of flood damage. 
        """

        raw_prob = self.model.predict_proba(data[self.features])[:,1]
        calibrated_prob = self.calibrator.predict(raw_prob)
        
        return(calibrated_prob)

### *** INITIAL SETUP *** ###

# Get current working directory 
pwd = os.getcwd()

# Get number of CPU cores
n_cores = int(os.environ['SLURM_NTASKS'])

# Get number of event to use for validation
validation_event_number = int(os.environ['SLURM_ARRAY_TASK_ID'])

# Get list of events to use for training (making sure to exclude those used for validation) 
training_event_numbers = np.arange(42,100+1)
training_event_numbers = training_event_numbers[training_event_numbers != validation_event_number]
included_event_numbers = np.concatenate((training_event_numbers,[validation_event_number]))

# Create output folder
outfolder = os.path.join(pwd,f'cross_validation/event_{validation_event_number:04d}')
if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)

# Get event information
event_catalog_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/event_delineation/historical_TC_event_info.csv'
event_catalog = pd.read_csv(event_catalog_path)
event_catalog['START_DATE'] = pd.to_datetime(event_catalog['START_DATE'])
event_catalog['END_DATE'] = pd.to_datetime(event_catalog['END_DATE'])

training_event_catalog = event_catalog[event_catalog['EVENT_NUMBER'].isin(training_event_numbers)]
validation_event_info = event_catalog[event_catalog['EVENT_NUMBER']==validation_event_number].iloc[0]

# (!) Remove once finalized (initially train on just the big ones)
training_event_catalog = training_event_catalog[training_event_catalog['NUM_CLAIMS']>= 500]
included_event_numbers = np.concatenate((training_event_catalog['EVENT_NUMBER'].to_numpy(),[validation_event_number])).tolist()

print('\n*** TROPICAL CYCLONE EVENT USED FOR VALIDATION ***\n')
print(validation_event_info,flush=True)

### *** LOAD DATA *** ###

## Stochastically-assigned presence-absence points derived from NFIP records

presence_absence_dir = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/nfip_building_matching/presence_absence_data/stochastic_assignment'
presence_absence_filepaths = [os.path.join(presence_absence_dir,f'event_{event_number:04d}_building_assignments.parquet') for event_number in included_event_numbers]

# Stochastic assignment was repeated multiple times
# Specify how many realizations to include 
num_replicates = 1
filters = [('replicate','<=',num_replicates)]

# Read in data as a dask dataframe 
presence_absence_data = dd.concat([dd.read_parquet(filepath,filters=filters) for filepath in presence_absence_filepaths]).reset_index(drop=True)
presence_absence_data = harmonize_dtypes(presence_absence_data)

## Structure attributes
# (Mainly interested in coordinates so that we can create spatial blocks for inner CV loop) 

included_counties_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/event_delineation/included_county_GEOIDs.txt'
included_counties = np.loadtxt(included_counties_path,dtype='str').tolist()

structure_info_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/nfip_building_matching/structure_info/CONUS_structure_info.parquet'
filters = [('countyfips_2010','in',included_counties)]
usecols = ['BUILD_ID','LATITUDE','LONGITUDE','x_epsg5070','y_epsg5070','countyfips_2010']
structure_info = dd.read_parquet(structure_info_path,columns=usecols,filters=filters)

# Specify size of blocks to use in spatial block CV, and create a unique block ID
block_size = 2000
structure_info['x_block'] = block_size*(structure_info['x_epsg5070'] / block_size).round()
structure_info['y_block'] = block_size*(structure_info['y_epsg5070'] / block_size).round()
structure_info['SPATIAL_BLOCK_ID'] = '(' + structure_info['x_block'].map('{:.1f}'.format,meta=('x_block','string[pyarrow]')) + ',' + structure_info['y_block'].map('{:.1f}'.format,meta=('y_block','string[pyarrow]')) + ')'
structure_info = structure_info.drop(columns=['x_block','y_block'])

structure_info = downcast_floats(structure_info)
structure_info = harmonize_dtypes(structure_info)

## Topographic features

included_RPUs = ['03a','03b','03c','03d','03e','03f'] # (!) expand once finalized. 
raster_dir = '/proj/characklab/projects/kieranf/flood_damage_index/data/rasters'
raster_filepaths = [os.path.join(raster_dir,f'{RPU}/{RPU}_raster_values_at_structure_points.parquet') for RPU in included_RPUs]
topo_data = dd.concat([dd.read_parquet(filepath) for filepath in raster_filepaths]).reset_index(drop=True)
topo_data['nhd_catchment_comid'] = topo_data['nhd_catchment_comid'].astype('int64[pyarrow]')
topo_data['cora_shoreline_node'] = topo_data['cora_shoreline_node'].astype('int64[pyarrow]')

topo_features = ['key_attributes_imputed',
                 'dist_coast_m',
                 'elev_cm',
                 'dist_wb_m',
                 'hand_wb_cm',
                 'geomorphon',
                 'tpi_cm']

topo_data = topo_data[['BUILD_ID','nhd_catchment_comid','cora_shoreline_node']+topo_features]
topo_data = downcast_floats(topo_data)
topo_data = harmonize_dtypes(topo_data)

## Precipitation intensity

precip_filepath = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/rainfall_runoff/precip_by_event/combined_precip_by_event.parquet'
filters = [('EVENT_NUMBER','in',included_event_numbers)]
precip_data = dd.read_parquet(precip_filepath,filters=filters).rename(columns={'comid':'nhd_catchment_comid'})

precip_features = ['C24_area_sqkm',
                   'C24_API120_mm',
                   'C24_MAI3_mmhr',
                   'C24_MAI6_mmhr',
                   'C24_MAI12_mmhr',
                   'C24_MAI24_mmhr',
                   'C24_MAI72_mmhr']

precip_data = precip_data[['EVENT_NUMBER','nhd_catchment_comid']+precip_features]
precip_data = downcast_floats(precip_data)
precip_data = harmonize_dtypes(precip_data)

## Storm surge

storm_surge_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/storm_surge/CORA_max_zeta_by_event/max_zeta_by_event.parquet'
storm_surge_data = dd.read_parquet(storm_surge_path).rename(columns={'nodenum':'cora_shoreline_node'})

storm_surge_features = ['max_zeta_over_threshold']

storm_surge_data = storm_surge_data[['EVENT_NUMBER','cora_shoreline_node']+storm_surge_features]
storm_surge_data = downcast_floats(storm_surge_data)
storm_surge_data = harmonize_dtypes(storm_surge_data)

### *** MERGE DATA *** ###

# Attach data on structure attributes (static)
presence_absence_data = dd.merge(presence_absence_data,structure_info,on='BUILD_ID',how='left')

# Attach data on topographic attributes (static)
presence_absence_data = dd.merge(presence_absence_data,topo_data,on='BUILD_ID',how='left')

# Attach data on precipitation intensity (dynamic) 
presence_absence_data = dd.merge(presence_absence_data,precip_data,on=['EVENT_NUMBER','nhd_catchment_comid'],how='left')

# Attach data on storm surge conditions (dynamic) 
presence_absence_data = dd.merge(presence_absence_data,storm_surge_data,on=['EVENT_NUMBER','cora_shoreline_node'],how='left')

# Specify whether any features represent categorical (as opposed to numeric) variables
categorical_features = ['geomorphon']

for feature in categorical_features:
    presence_absence_data[feature] = presence_absence_data[feature].astype('category')

# Assemble dataset into local memory as a pandas dataframe 
presence_absence_data = presence_absence_data.compute()

### *** CROSS VALIDATION *** ###

# Get list of features to use in model
features = topo_features + precip_features + storm_surge_features

# Specify hyperparameters
hyperparams = {'objective':'binary',
               'learning_rate':0.05,
               'n_estimators':2000,
               'num_leaves':31,
               'max_depth':6,
               'min_child_samples':200,
               'reg_lambda':1.0,
               'subsample':0.8,
               'subsample_freq':1,
               'colsample_bytree':0.8,
               'max_bin':255,
               'min_data_per_group':200,
               'cat_smooth':10,
               'n_jobs':n_cores}

# Split into train / test set
test_mask = (presence_absence_data['EVENT_NUMBER'] == validation_event_number)
train_mask = ~test_mask

# Initialize model 
mod = FloodDamageProbabilityEstimator(hyperparams,
                                      'flooded',
                                      features,
                                      categorical_features=categorical_features,
                                      block_id_col='SPATIAL_BLOCK_ID')

# Fit model to training data
mod.fit_probabilities(presence_absence_data[train_mask])

# Apply to testing data
test_data = presence_absence_data[test_mask]
test_data[mod.outcome_prob_variable] = mod.predict_probabilities(test_data)

### *** SAVE RESULTS *** ###

# Test data 
outname = os.path.join(outfolder,f'event_{validation_event_number:04d}_sampled_test_data.parquet')
test_data.to_parquet(outname)

# Fitted FloodDamageProbabilityEstimator 
outname = os.path.join(outfolder,f'event_{validation_event_number:04d}_FloodDamageProbabilityEstimator.pickle')
with open(outname,'wb') as f:
    pickle.dump(mod,f)
    f.close()