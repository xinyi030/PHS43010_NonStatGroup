import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os 
import warnings 
from run_logistic_simulations import run_single_trial
RANDOM_SEED = 58
rng = np.random.default_rng(RANDOM_SEED)
warnings.filterwarnings("ignore")

parquet_file_name = "S1_finished_simulations.parquet"
pickle_file_name = "S1_sigmoid_probs.pkl"


doses = [0.5, 1, 3, 5, 6] # From figure 6 and in units (mg/m^2 per day)
true_toxic_prob = (0.25, 0.3, 0.5, 0.6, 0.7) # Given by assignment instructions, scenario 1

total_data = []
sigmoid_probabilities = []
# we should have 1000 simulations in total
for sim in tqdm(range(500)):
    data, trial_probabilities = run_single_trial()
    total_data.append(data)    
    sigmoid_probabilities.append(trial_probabilities)
    
# save the sigmoid probabilities to a pickle file
with open(pickle_file_name, "wb") as f: # "wb" because we want to write in binary mode
    pickle.dump(sigmoid_probabilities, f)
    
    
for i in range(len(total_data)):
    # add a column that tracks which simulation run is associated with each row
    total_data[i]['sim_run'] = i
    
# combine list of dataframes to one dataframe
total_data = pd.concat(total_data) 
total_data.to_parquet(f'gabe_sim_files/{parquet_file_name}')