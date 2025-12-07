import numpy as np
import pandas as pd
import os

pwd = os.getcwd()

state_list_path = '/proj/characklab/projects/kieranf/flood_damage_index/analysis/CONUS_state_list.txt'
state_list = np.loadtxt(state_list_path,dtype=str)

filepaths = [os.path.join(pwd,'structure_info',state,f'{state}_structure_info.parquet') for state in state_list]

df = pd.concat([pd.read_parquet(f) for f in filepaths]).reset_index(drop=True)

outname = os.path.join(pwd,'structure_info/CONUS_structure_info.parquet')
df.to_parquet(outname)
