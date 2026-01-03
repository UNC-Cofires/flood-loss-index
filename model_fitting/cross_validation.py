import numpy as np
import pandas as pd
import scipy.stats as stats
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
import pickle
import time
import os

### *** HELPER FUNCTIONS *** ###

# Helper function to format elapsed time in seconds
def format_elapsed_time(seconds):
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

def normalize_weights(df,id_col='record_id',weight_col='weight',scale_col=None):
    """
    This function normalizes the weights assigned to potential building matches so that 
    the weights associated with each insurance record sum to 1.0. 
    
    param: df: dataframe where each row represents a potential record-building matchhis 
    param: id_col: name of column containing unique ID of each record. Can also be a list of column names. 
    param: weight_col: name of column containing weight assigned to each building
    param: scale_col: name of column containing values to scale normalized weight by. 
                     (this allows us to normalize weights as if each group contained only 
                     one record, and then can multiply weights by number of records in each 
                     matching group.) 
    """
    
    df[weight_col] = df.groupby(id_col)[weight_col].transform(lambda x: x / x.sum())

    if scale_col is not None:
        df[weight_col] *= df[scale_col]
    
    return(df)

### *** PROBABILITY CALIBRATION CLASS *** ###

class PlattScalingCalibrator:
    """
    Helper class that uses Platt scaling to calibrate raw probability estimates of ML models
    """
    
    def __init__(self,beta0=0,beta1=1):
        self.beta0 = beta0
        self.beta1 = beta1

    def fit(self,p,y,sample_weight=None,smallnum=1e-9):

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
        x = safe_logit(raw_prob)
        calibrated_prob = 1/(1+np.exp(-1*(self.beta0 + self.beta1*x)))
        return calibrated_prob

### *** FLOOD DAMAGE PREDICTION CLASS *** ###

class FloodDamageProbabilityEstimator:

    def __init__(self,
                 candidate_df,
                 outcome_variable,
                 features,
                 hyperparams,
                 record_id_col='RECORD_ID',
                 building_id_col='BUILD_ID',
                 block_id_col='SPATIAL_BLOCK_ID',
                 weight_col='weight',
                 prior_col='prior',
                 scale_col=None):
        """
        param: candidate_df: building candidate data used for training [pandas dataframe]. Each row
                        should represent a potential match between an insurance record (or group of 
                        insurance records) and a building. By aggregating together insurance records 
                        that share the same outcome and building candidates, we can save on memory and 
                        computation time; however, doing this will require us to scale weights by the 
                        number of records in each group. For more info, see notes on scale_col parameter. 
                        
        param: outcome_variable: name of binary variable representing presence or absence of flood damage [string]. 
        
        param: features: list of variables used to predict outcome [list of strings]. Please note that 
                        categorical features should be saved as the pandas "category" dtype in the 
                        train_candidate_df and test_candidate_df dataframes. 
                         
        param: hyperparams: dictionary of hyperparameters to be used by LGBMClassifier [dict]. 
        
        param: record_id_col: name of variable that uniquely identifies each insurance record or group of 
                        aggregated insurance records [string]. 
                              
        param: building_id_col: name of variable that uniquely identifies each building record. 
        
        param: block_id_col: name of variable that identifies the spatial and/or temporal block that 
                        each building belongs to [string]. When fitting the model and calibrating 
                        probabilites, we'll do an inner cross-validation loop that utilizes blocked 
                        cross-validation. This helps to overcome issues related to spatial and/or 
                        temporal autocorrelation in the data. For more info, see doi:10.1111/ecog.02881. 

        param: weight_col: name of variable corresponding to weight given to each building candidate. 
                        This weight represents the probability that a specific insurance record was 
                        generated by a specific building candidate. These weights will be iteratively
                        updated using the EM algorithm. 

        param: prior_col: name of variable representing the prior probability that a building candidate
                        generated a specific insurance record before observing the outcome of that 
                        insurance record (e.g., flooded or not flooded) [string]. Typically, our prior 
                        belief will be that all building candidates are equally likely (uniform prior); 
                        however, we could change this assumption by adjusting our priors to account 
                        for insurance purchase behaviors (e.g., shabby houses less likely to be insured). 

        param: scale_col: name of variable used to scale weights calculated for each building candidate [string]. 
                        If similar insurance records have been aggregated together to save on memory and 
                        computation time, then this should be equal to the number of records in each group.
                        Otherwise, this should be set to none. 
        """

        self.candidate_df = candidate_df
        self.outcome_variable = outcome_variable
        self.outcome_prob_variable = f'{outcome_variable}_prob'
        self.features = features

        # Check that features and outcome variable are included in train and test data
        check1 = all(f in self.candidate_df.columns for f in self.features)
        check2 = self.outcome_variable in self.candidate_df.columns

        if not check1&check2:
            raise ValueError('Features and/or outcome variable are missing from the training data.')

        # Get list of categorical features
        self.feature_dtypes = self.candidate_df.dtypes[self.features]
        self.categorical_features = self.feature_dtypes[self.feature_dtypes=='category'].index.to_list()

        # Save LGBMClassifier hyperparams
        self.hyperparams = hyperparams
        self.model = None
        self.calibrator = None
        
        # Check that parameters related to spatial blocking and candidate weighting are in dataset
        param_names = ['Record ID','Building ID','Block ID','Weight','Prior']
        param_values = [record_id_col,building_id_col,block_id_col,weight_col,prior_col]

        if scale_col is not None:
            param_names += ['Scale']
            param_values += [scale_col]

        for name,user_input in zip(param_names,param_values):
            if user_input not in self.candidate_df.columns:
                raise ValueError(f'{name} column \'{user_input}\' not found in training data.')
        
        # Now that we've checked the user input, save parameters related to spatial 
        # blocking and candidate weighting.
        self.record_id_col = record_id_col
        self.building_id_col = building_id_col
        self.block_id_col = block_id_col
        self.weight_col = weight_col
        self.prior_col = prior_col
        self.scale_col = scale_col

        # Normalize candidate weights once at start in case the user hasn't already done this
        self.candidate_df = normalize_weights(self.candidate_df,id_col=self.record_id_col,weight_col=self.weight_col,scale_col=self.scale_col)

        return(None)

    def fit_probabilities(self,num_samples=2000000,num_calibration_folds=5,t0=None):
        """
        This function fits an LGBMClassifier model to the data while using an inner
        cross-validation loop to calibrate flood damage probability projections. 

        param: num_samples: number of samples to draw from candidate_df [integer]. Each
                        sample can be thought of as a monte-carlo assignment of an 
                        insurance record to a building candidate. Samples are drawn 
                        according to building candidate weights (i.e., more likely 
                        candidates are more likely to be selected). The total amount
                        of computation time required to fit the model depends strongly 
                        on this parameter. 
        param: num_calibration_folds: number of inner cross-validation folds to use when 
                        calibrating predicted probabilities to data. 
        param: t0: time since start of process [float]. Encoded as # seconds since Unix epoch. 
                        Users can pass this as an argument if the function call is part of a 
                        larger workflow. 
        """
        
        if t0 is None:
            t0 = time.time()

        # Samples from buildings candidates according to candidate weight 
        train_df = self.candidate_df.sample(num_samples,replace=True,weights=self.weight_col).reset_index(drop=True)

        # Set up inner CV loop that we'll use to calibrate probabilites 
        spatial_folds = pd.DataFrame(train_df[self.block_id_col].unique(),columns=[self.block_id_col])
        spatial_folds['fold'] = np.random.randint(num_calibration_folds,size=len(spatial_folds))
        train_df = pd.merge(train_df,spatial_folds,on=self.block_id_col)
        
        train_df[self.outcome_prob_variable] = np.nan

        # Within each loop, keep track of n_estimators. Important 
        # for preventing overfitting when we give the model all the data. 
        best_iteration_list = []

        # CV loop 
        for fold in range(num_calibration_folds):

            t1 = time.time()
        
            print(f'\n\n*** CALIBRATION FOLD {fold+1} / {num_calibration_folds} ***\n\n')
            
            fold_mask = (train_df['fold']==fold)
            
            model = lgb.LGBMClassifier(**self.hyperparams)
            
            callbacks=[lgb.early_stopping(stopping_rounds=50,min_delta=1e-5),
                       lgb.log_evaluation(period=50)]
            
            model.fit(train_df[~fold_mask][self.features],
                      train_df[~fold_mask][self.outcome_variable],
                      feature_name=self.features,
                      categorical_feature=self.categorical_features,
                      eval_set=[(train_df[fold_mask][self.features], train_df[fold_mask][self.outcome_variable])],
                      eval_metric="binary_logloss",
                      callbacks=callbacks)
        
            best_iteration_list.append(model.best_iteration_)
            
            train_df.loc[fold_mask,self.outcome_prob_variable] = model.predict_proba(train_df[fold_mask][features])[:,1]

            t2 = time.time()
            
            elapsed_time = format_elapsed_time(t2-t1)
            cumulative_elapsed_time = format_elapsed_time(t2-t0)

            print(f'\nTIME ELAPSED: {elapsed_time} last iteration / {cumulative_elapsed_time} cumulative')
        
        print(f'\n\n*** CALIBRATING PROBABILITIES ***\n\n',flush=True)

        t1 = time.time()
        
        # Use platt scaling to correct for systemic 
        # over- or underestimation of probabilities
        calibrator = PlattScalingCalibrator()
        calibrator.fit(train_df[self.outcome_prob_variable],train_df[self.outcome_variable])
        self.calibrator = calibrator

        print(f'Platt scaling parameters: beta0 = {calibrator.beta0:.3f}, beta1 = {calibrator.beta1:.3f}',flush=True)

        t2 = time.time()

        elapsed_time = format_elapsed_time(t2-t1)
        cumulative_elapsed_time = format_elapsed_time(t2-t0)

        print(f'\nTIME ELAPSED: {elapsed_time} last iteration / {cumulative_elapsed_time} cumulative')
        
        print(f'\n\n*** POST-CALIBRATION MODEL FITTING ***\n\n',flush=True)

        t1 = time.time()
        
        # Now fit to all the data, and use this to create calibrated probabilities
        n_estimators = int(np.median(best_iteration_list))
        model = lgb.LGBMClassifier(**self.hyperparams)
        model.n_estimators = n_estimators
        
        model.fit(train_df[self.features],
                  train_df[self.outcome_variable],
                  feature_name=self.features,
                  categorical_feature=self.categorical_features)

        self.model = model

        t2 = time.time()

        elapsed_time = format_elapsed_time(t2-t1)
        cumulative_elapsed_time = format_elapsed_time(t2-t0)

        print(f'\nTIME ELAPSED: {elapsed_time} last iteration / {cumulative_elapsed_time} cumulative')

        return(None)

    def predict_probabilities(self,data):
        """
        param: data: data on which to to predict probabilities [pandas dataframe]. Must follow same 
                        format as candidate_df. 
        returns: calibrated_prob: predicted probability of flood damage. 
        """

        raw_prob = self.model.predict_proba(data[self.features])[:,1]
        calibrated_prob = self.calibrator.predict(raw_prob)
        
        return(calibrated_prob)

    def EM_iteration(self,num_samples=2000000,t0=None):
        """
        This function updates building candidate weights and model parameters
        using the expectation-maximization (EM) algorithm. 

        param: num_samples: number of samples to draw from candidate_df [integer]. Each
                        sample can be thought of as a monte-carlo assignment of an 
                        insurance record to a building candidate. Samples are drawn 
                        according to building candidate weights (i.e., more likely 
                        candidates are more likely to be selected). The total amount
                        of computation time required to fit the model depends strongly 
                        on this parameter. 
        param: t0: time since start of process [float]. Encoded as # seconds since Unix epoch. 
                        Users can pass this as an argument if the function call is part of 
                        a larger workflow. 
        """

        ## Initialization
        if t0 is None:
            t0 = time.time()

        # Small number to avoid machine precision issues
        smallnum=1e-8

        # Fit model using starting values of weights
        # (Can skip this step if the model has already been fit) 
        if self.model is None:
            self.fit_probabilities(num_samples=num_samples,t0=t0)

        # Get prior probabilities
        prior = self.candidate_df[self.prior_col].to_numpy()

        ## E-step: Update candidate weights

        print('\n\n### --------- START OF EM ITERATION --------- ###\n\n',flush=True)

        t1 = time.time()
    
        # Compute likelihood of observed outcomes at each building candidate
        # based on current values of model parameters. 
        y = self.candidate_df[self.outcome_variable]
        p = self.predict_probabilities(self.candidate_df)        
        likelihood = stats.bernoulli.pmf(y,p)
    
        # Update weights using Bayes rule (Posterior ~ Likelihood x Prior)
        # Add tiny number to avoid issues related to machine precision
        self.candidate_df[self.weight_col] = likelihood*prior + smallnum
        self.candidate_df = normalize_weights(self.candidate_df,id_col=self.record_id_col,weight_col=self.weight_col,scale_col=self.scale_col)

        t2 = time.time()

        estep_elapsed_time = format_elapsed_time(t2-t1)
        
        ## M-step: Update model parameters

        t1 = time.time()
    
        # Re-fit the model using new weights
        self.fit_probabilities(num_samples=num_samples,t0=t0)

        t2 = time.time()

        mstep_elapsed_time = format_elapsed_time(t2-t1)
        cumulative_elapsed_time = format_elapsed_time(t2-t0)

        print('\n\n### --------- END OF EM ITERATION --------- ###',flush=True)
        print(f'\nTIME ELAPSED: {estep_elapsed_time} E-step / {mstep_elapsed_time} M-step / {cumulative_elapsed_time} cumulative')

        return(None)

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

# (!) Remove once debugged (initially train on just the big ones)
training_event_catalog = training_event_catalog[training_event_catalog['NUM_CLAIMS']>= 500]
included_event_numbers = np.concatenate((training_event_catalog['EVENT_NUMBER'].to_numpy(),[validation_event_number]))

print('\n*** TROPICAL CYCLONE EVENT USED FOR VALIDATION ***\n')
print(validation_event_info,flush=True)

### *** LOAD DATA *** ###

## Presence-absence information derived from anonymized NFIP records
presence_absence_data_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/nfip_building_matching/presence_absence_data/presence_absence_summary.parquet'
filters = [('EVENT_NUMBER','in',included_event_numbers)]
presence_absence_data = pd.read_parquet(presence_absence_data_path,filters=filters)
included_match_keys = presence_absence_data['match_key'].unique()

## Building candidates
building_lookup_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/nfip_building_matching/potential_matches/CONUS_nfip_matching_building_lookup.parquet'
filters = [('match_key','in',included_match_keys)]
building_lookup = pd.read_parquet(building_lookup_path,filters=filters)

## Topographic features
included_RPUs = ['03a','03b','03c','03d','03e','03f'] # (!) expand once debugged. 
raster_dir = '/proj/characklab/projects/kieranf/flood_damage_index/data/rasters'
raster_filepaths = [os.path.join(raster_dir,f'{RPU}/{RPU}_raster_values_at_structure_points.parquet') for RPU in included_RPUs]
topo_data = pd.concat([pd.read_parquet(filepath) for filepath in raster_filepaths]).reset_index(drop=True)
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

## Precipitation intensity
precip_filepath = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/rainfall_runoff/precip_by_event/combined_precip_by_event.parquet'
filters = [('EVENT_NUMBER','in',included_event_numbers)]
precip_data = pd.read_parquet(precip_filepath,filters=filters)
precip_data.rename(columns={'comid':'nhd_catchment_comid'},inplace=True)

precip_features = ['C24_area_sqkm',
                   'C24_API120_mm',
                   'C24_MAI3_mmhr',
                   'C24_MAI6_mmhr',
                   'C24_MAI12_mmhr',
                   'C24_MAI24_mmhr',
                   'C24_MAI48_mmhr',
                   'C24_MAI72_mmhr']

precip_data = precip_data[['EVENT_NUMBER','nhd_catchment_comid']+precip_features]

## Storm surge

storm_surge_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/storm_surge/CORA_max_zeta_by_event/max_zeta_by_event.parquet'
storm_surge_data = pd.read_parquet(storm_surge_path).rename(columns={'nodenum':'cora_shoreline_node'})

storm_surge_features = ['max_zeta_over_threshold']

storm_surge_data = storm_surge_data[['EVENT_NUMBER','cora_shoreline_node']+storm_surge_features]

## Structure coordinates 

structure_info_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/nfip_building_matching/structure_info/CONUS_structure_info.parquet'
filters = [('state','in',building_lookup['state'].unique())]
usecols = ['BUILD_ID','LATITUDE','LONGITUDE','x_epsg5070','y_epsg5070']
structure_info = pd.read_parquet(structure_info_path,columns=usecols,filters=filters)

# Specify size of blocks to use in spatial block CV, and create a unique block ID
block_size = 2000
structure_info['x_block'] = np.round(structure_info['x_epsg5070'] / block_size)*block_size
structure_info['y_block'] = np.round(structure_info['y_epsg5070'] / block_size)*block_size
structure_info['SPATIAL_BLOCK_ID'] = '(' + structure_info['x_block'].map('{:.1f}'.format) + ',' + structure_info['y_block'].map('{:.1f}'.format) + ')'
structure_info.drop(columns=['x_block','y_block'],inplace=True)

# Attach information to building lookup table
building_lookup = pd.merge(building_lookup,structure_info,on='BUILD_ID',how='left')

### *** PREPROCESS DATA *** ###

## Create dataframe listing potential outcomes of each building candidate

# Make sure identifier columns are encoded correctly
presence_absence_data['match_key'] = presence_absence_data['match_key'].astype('string[pyarrow]')
building_lookup['match_key'] = building_lookup['match_key'].astype('string[pyarrow]')

# Create a unique identifier representing group used to create presence absence counts
presence_absence_data['PA_GROUP_ID'] = np.arange(len(presence_absence_data))+1

# Change format of data from wide to long. 
# Create indicator denoting positive (flood presence) and negative (flood absence) potential outcomes. 
positive_outcomes = presence_absence_data[['EVENT_NUMBER','PA_GROUP_ID','match_key','num_buildings','num_presence']].rename(columns={'num_presence':'num_records'})
positive_outcomes['flooded'] = 1
negative_outcomes = presence_absence_data[['EVENT_NUMBER','PA_GROUP_ID','match_key','num_buildings','num_absence']].rename(columns={'num_absence':'num_records'})
negative_outcomes['flooded'] = 0

candidate_df = pd.concat([positive_outcomes,negative_outcomes]).sort_values(by=['EVENT_NUMBER','PA_GROUP_ID','flooded'])

# Drop potential outcomes that have no associated insurance records
# (The "weight" given to these will always be zero, so no point to include them) 
candidate_df = candidate_df[candidate_df['num_records'] > 0].reset_index(drop=True)

# Create unique identifier denoting unique record groups 
# (e.g., flooded insurance records from a specific census block + storm) 
candidate_df['RECORD_GROUP_ID'] = candidate_df['PA_GROUP_ID'].astype(str) + '_' + candidate_df['flooded'].astype(str)

# Add building information
candidate_df = pd.merge(candidate_df,building_lookup[['match_key','BUILD_ID','LATITUDE','LONGITUDE','SPATIAL_BLOCK_ID']],on='match_key',how='left')
candidate_df = candidate_df[['EVENT_NUMBER','PA_GROUP_ID','RECORD_GROUP_ID','BUILD_ID','LATITUDE','LONGITUDE','SPATIAL_BLOCK_ID','match_key','num_buildings','num_records','flooded']]

# Add columns that we'll use to calculate weight given to soft labels
# (these will be used by EM algorithm to adjust weights given to building 
# candidates conditional on the observed outcome of insurance records) 
candidate_df['prior'] = 1/candidate_df['num_buildings']
candidate_df['likelihood'] = np.nan
candidate_df['weight'] = candidate_df['prior'].copy() # Set initial value of weights equal to prior

# Normalize weights so that sum of building weights within each candidate group is equal to 
# the number of insurance records associated with that candidate group
# (equivalent to saying that candidate weights for each record must sum to 1.0) 
candidate_df = normalize_weights(candidate_df,id_col='RECORD_GROUP_ID',weight_col='weight',scale_col='num_records')

## Add features to dataframe 

# Make sure identifier columns are encoded correctly and consistently
candidate_df['EVENT_NUMBER'] = candidate_df['EVENT_NUMBER'].astype('int64[pyarrow]')
candidate_df['BUILD_ID'] = candidate_df['BUILD_ID'].astype('string[pyarrow]')

topo_data['BUILD_ID'] = topo_data['BUILD_ID'].astype('string[pyarrow]')
topo_data['nhd_catchment_comid'] = topo_data['nhd_catchment_comid'].astype('int64[pyarrow]')
topo_data['cora_shoreline_node'] = topo_data['cora_shoreline_node'].astype('int64[pyarrow]')

precip_data['EVENT_NUMBER'] = precip_data['EVENT_NUMBER'].astype('int64[pyarrow]')
precip_data['nhd_catchment_comid'] = precip_data['nhd_catchment_comid'].astype('int64[pyarrow]')

storm_surge_data['EVENT_NUMBER'] = storm_surge_data['EVENT_NUMBER'].astype('int64[pyarrow]')
storm_surge_data['cora_shoreline_node'] = storm_surge_data['cora_shoreline_node'].astype('int64[pyarrow]')

# Attach data on topographic attributes (static)
candidate_df = pd.merge(candidate_df,topo_data,on='BUILD_ID',how='left')

# Attach data on precipitation intensity (dynamic) 
candidate_df = pd.merge(candidate_df,precip_data,on=['EVENT_NUMBER','nhd_catchment_comid'],how='left')

# Attach data on storm surge conditions (dynamic) 
candidate_df = pd.merge(candidate_df,storm_surge_data,on=['EVENT_NUMBER','cora_shoreline_node'],how='left')

## Get list of features to use in model
features = topo_features + precip_features + storm_surge_features

# Specify categorical features
categorical_features = ['geomorphon']
for feature in categorical_features:
   candidate_df[feature] = candidate_df[feature].astype('category')

# Downcast float64 to float32 to save on memory
feature_data_types = candidate_df[features].dtypes
for feature in feature_data_types[feature_data_types == 'float64'].index.values:
    candidate_df[feature] = candidate_df[feature].astype('float32')

### *** CROSS VALIDATION *** ###

hyperparams = {'objective':'binary',
               'learning_rate':0.05,
               'n_estimators':1000,
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

test_mask = (candidate_df['EVENT_NUMBER'] == validation_event_number)
train_mask = ~test_mask

mod = FloodDamageProbabilityEstimator(candidate_df[train_mask],
                                      'flooded',
                                      features,
                                      hyperparams,
                                      record_id_col='RECORD_GROUP_ID',
                                      building_id_col='BUILD_ID',
                                      block_id_col='SPATIAL_BLOCK_ID',
                                      weight_col='weight',
                                      prior_col='prior',
                                      scale_col='num_records')

mod.fit_probabilities(num_samples=50000000)

test_candidate_df = candidate_df[test_mask].copy()
test_candidate_df[mod.outcome_prob_variable] = mod.predict_probabilities(test_candidate_df)

### *** SAVE RESULTS *** ###

# Data
outname = os.path.join(outfolder,f'event_{validation_event_number:04d}_test_data.parquet')
test_candidate_df.to_parquet(outname)

# Fitted LGBMClassifier 
outname = os.path.join(outfolder,f'event_{validation_event_number:04d}_LightGBM_model.pickle')
with open(outname,'wb') as f:
    pickle.dump(mod.model,f)
    f.close()

# Fitted calibrator 
outname = os.path.join(outfolder,f'event_{validation_event_number:04d}_calibrator.pickle')
with open(outname,'wb') as f:
    pickle.dump(mod.calibrator,f)
    f.close()