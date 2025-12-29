import numpy as np
import pandas as pd
import os

pwd = os.getcwd()
precip_dir = os.path.join(pwd,'precip_by_event')
filenames = np.sort([x for x in os.listdir(precip_dir) if x.startswith('event')])

def process_precip_data(filename):
    
    filepath = os.path.join(precip_dir,filename)
    event_number = int(filename.split('_')[1])
    
    df = pd.read_parquet(filepath)
    df['EVENT_NUMBER'] = event_number

    return(df)

combined_precip_data = pd.concat([process_precip_data(filename) for filename in filenames]).reset_index(drop=True)

outname = os.path.join(precip_dir,'combined_precip_by_event.parquet')
combined_precip_data.to_parquet(outname)