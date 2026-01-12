import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import time
import os

### *** HELPER FUNCTIONS *** ###

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

### *** FLOOD DAMAGE PREDICTION CLASS *** ###

class FloodDamageRegressor:
    
    def __init__(self,
                 hyperparams,
                 outcome_variable,
                 features,
                 weight_col,
                 block_id_col,
                 categorical_features=[]):
        """
        param: hyperparams: dictionary of hyperparameters to be used by LGBMRegressor [dict]. 
        param: outcome_variable: name of variable representing claim intensity (# claims / # policies) [string].
        param: features: list of variables used to predict outcome [list of strings].
        param: weight_col: name of variable denoting weight to give to each training example [string].
                        This is useful for dealing with the small numbers problem, as we will likely want 
                        to weight claim intensities by the number of policies in each unit. 
        param: block_id_col: name of variable that identifies the spatial and/or temporal block that 
                        each building belongs to [string]. When fitting the model and determining the 
                        number of trees, we'll do an inner cross-validation loop that utilizes blocked 
                        cross-validation. This helps to overcome issues related to spatial and/or 
                        temporal autocorrelation in the data. For more info, see doi:10.1111/ecog.02881. 
        param: categorical_features: list of variables that are categorical [list of strings]. When fitting 
                        and applying models to data, these variables should be cast as the pandas "category" 
                        data type. 
         """

        # Save information on outcome variable and features
        self.outcome_variable = outcome_variable
        self.outcome_pred_variable = f'{outcome_variable}_pred'
        self.features = features
        self.weight_col = weight_col
        self.block_id_col = block_id_col
        self.categorical_features = categorical_features

        # Save LGBMRegressor hyperparams
        self.hyperparams = hyperparams
        self.model = None

    def fit(self,data,num_calibration_folds=5,min_delta=1e-6,t0=None):
        """
        This function fits an LGBMRegressor model to the data while using an inner
        cross-validation loop to calibrate claim intensity projections. 

        param: data: pandas dataframe containing data used for model fitting. Must contain 
                        columns corresponding to outcome_variable, features, and 
                        block_id_col attributes. 
        param: num_calibration_folds: number of inner cross-validation folds to use when 
                        determining number of trees to fit. 
        param: min_delta: minimum incremental change in objective function to justify adding
                        more trees (used within early stopping). 
        param: t0: time since start of process [float]. Encoded as # seconds since Unix epoch. 
                        Users can pass this as an argument if the function call is part of a 
                        larger workflow. 
        """

        # If the user hasn't already done so, start counting time. 
        if t0 is None:
            t0 = time.time()

        # Set up inner spatial and/or temporal block CV loop that we'll use to tune n_estimators
        block_cv_folds = pd.DataFrame(data[self.block_id_col].unique(),columns=[self.block_id_col])
        block_cv_folds['fold'] = np.random.randint(num_calibration_folds,size=len(block_cv_folds))
        data = pd.merge(data,block_cv_folds,on=self.block_id_col,how='left')

        # Initialize column that we'll use to keep track of predicted claim intensities
        data[self.outcome_pred_variable] = np.nan

        # Within each loop, keep track of n_estimators. 
        # This is important for preventing overfitting in final step 
        # where we fit to all the data. 
        best_iteration_list = []

        # Inner CV loop 
        for fold in range(num_calibration_folds):

            t1 = time.time()
        
            print(f'\n\n*** CALIBRATION FOLD {fold+1} / {num_calibration_folds} ***\n',flush=True)
            
            fold_mask = (data['fold']==fold)
            
            model = lgb.LGBMRegressor(**self.hyperparams)
            
            callbacks=[lgb.early_stopping(stopping_rounds=100,min_delta=min_delta),
                       lgb.log_evaluation(period=100)]
            
            model.fit(data[~fold_mask][self.features],
                      data[~fold_mask][self.outcome_variable],
                      sample_weight=data[~fold_mask][self.weight_col],
                      feature_name=self.features,
                      categorical_feature=self.categorical_features,
                      eval_set=[(data[fold_mask][self.features], data[fold_mask][self.outcome_variable])],
                      eval_sample_weight=[data[fold_mask][self.weight_col]],
                      eval_metric='regression',
                      callbacks=callbacks)
        
            best_iteration_list.append(model.best_iteration_)
            
            data.loc[fold_mask,self.outcome_pred_variable] = model.predict(data[fold_mask][self.features])

            t2 = time.time()
            
            elapsed_time = format_elapsed_time(t2-t1)
            cumulative_elapsed_time = format_elapsed_time(t2-t0)

            print(f'\nTIME ELAPSED: {elapsed_time} last iteration / {cumulative_elapsed_time} cumulative',flush=True)
        
        print(f'\n\n*** POST-CALIBRATION MODEL FITTING ***\n',flush=True)

        t1 = time.time()
        
        # Now fit to all the data
        n_estimators = int(np.mean(best_iteration_list))
        model = lgb.LGBMRegressor(**self.hyperparams)
        model.n_estimators = n_estimators
        
        model.fit(data[self.features],
                  data[self.outcome_variable],
                  sample_weight=data[self.weight_col],
                  feature_name=self.features,
                  categorical_feature=self.categorical_features)

        self.model = model

        t2 = time.time()

        elapsed_time = format_elapsed_time(t2-t1)
        cumulative_elapsed_time = format_elapsed_time(t2-t0)

        print(f'\nTIME ELAPSED: {elapsed_time} last iteration / {cumulative_elapsed_time} cumulative',flush=True)

        return(None)

    def predict(self,data):
        """
        param: data: data on which to to predict claim intensities [pandas dataframe]. Must follow same 
                        format as data used to fit model. 
        returns: y_pred: predicted claim intensity. 
        """

        y_pred = self.model.predict(data[self.features])
        
        return(y_pred)

### *** INITIAL SETUP *** ###

# Get current working directory 
pwd = os.getcwd()

# Get number of CPU cores
n_cores = int(os.environ['SLURM_NTASKS'])

# Get number of event to use for validation
validation_event_number = int(os.environ['SLURM_ARRAY_TASK_ID'])

# Get list of events to use for training (making sure to exclude those used for validation) 
training_event_numbers = np.arange(1,100+1)
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

print('\n*** TROPICAL CYCLONE EVENT USED FOR VALIDATION ***\n')
print(validation_event_info,flush=True)

### *** LOAD DATA *** ###

data_dir = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/model_fitting/training_data'
filepaths = [os.path.join(data_dir,f'event_{event_number:04d}_training_data.parquet') for event_number in included_event_numbers]
df = pd.read_parquet(filepaths)

## Get list of features to use in model

# Topographic variables  
topo_features = ['dist_coast_m',
                 'dist_wb_m',
                 'elev_cm',
                 'hand_wb_cm']

# Precipitation-related variables
precip_features = ['C72_area_sqkm',
                   'C72_API120_mm',
                   'C72_MAI24_mmhr',
                   'C0_MAI24_mmhr']

# Storm surge related variables
surge_features = ['max_zeta_over_threshold']

# Combine different groups of predictors into one list
features = topo_features + precip_features + surge_features

### *** CROSS VALIDATION *** ###

# Specify hyperparameters
hyperparams = {'objective':'regression',
               'learning_rate':0.01,
               'n_estimators':2000,
               'num_leaves':5,
               'max_depth':4,
               'min_child_samples':1000,
               'reg_alpha':1.0,
               'reg_lambda':10.0,
               'min_split_gain':0.1,
               'subsample':0.7,
               'subsample_freq':1,
               'colsample_bytree':0.7,
               'n_jobs':n_cores}

# Split into train / test set
test_mask = (df['EVENT_NUMBER'] == validation_event_number)
train_mask = ~test_mask
train_data = df[train_mask]
test_data = df[test_mask]


# Initialize model 
mod = FloodDamageRegressor(hyperparams,
                           'claim_intensity',
                           features,
                           'num_records',
                           'SPATIAL_BLOCK_ID')

# Fit model to training data
mod.fit(train_data,num_calibration_folds=5)

# Apply to testing data
test_data[mod.outcome_pred_variable] = mod.predict(test_data)

### *** SAVE RESULTS *** ###

# Test data 
outname = os.path.join(outfolder,f'event_{validation_event_number:04d}_test_data.parquet')
test_data.to_parquet(outname)

# Fitted FloodDamageProbabilityEstimator 
outname = os.path.join(outfolder,f'event_{validation_event_number:04d}_FloodDamageRegressor.pickle')
with open(outname,'wb') as f:
    pickle.dump(mod,f)
    f.close()
