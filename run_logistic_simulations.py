import numpy as np
import pandas as pd
import pymc as pm
from tqdm import tqdm
import pickle
import os 

RANDOM_SEED = 58
rng = np.random.default_rng(RANDOM_SEED)

doses = [0.5, 1, 3, 5, 6] # From figure 6 and in units (mg/m^2 per day)
true_toxic_prob_s1 = (0.25, 0.3, 0.5, 0.6, 0.7) # Given by assignment instructions, scenario 1
true_toxic_prob_s2 = (0.01, 0.05, 0.2, 0.3, 0.5) # Given by assignment instructions, scenario 2

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def sim_data(relevant_index):
    """ Simulates whether any of the 3 participants have a toxicity event given by the unknown probabilities.

    Args:
        relevant_index (Int): The index associated with the dose being used on the three samples.

    Returns:
        Pandas Dataframe: Contains the doses the three participants took and whether each participant had a toxicity event
    """
    toxicity_observation = np.random.binomial(1, p=true_toxic_prob_s1[relevant_index], size=3) # 3 samples per cohort
    doses_observed = np.ones(3) * doses[relevant_index] # doses given to the three participants
    # store data in a dataframe that will later be concatenated to make one dataframe
    data = pd.DataFrame({'doses':doses_observed, 'toxicity_event': toxicity_observation})
    return data


def run_model(data):
    """ Creates a logistic regression model using the priors given in the paper
    and the data collected up until this point. 

    Args:
        data (Pandas Dataframe): Same dataframe given by sim_data()

    Returns:
        pymc model: Fitted model based on priors and the current data collected
    """
    coords = {"observation": data.index.values} 
    with pm.Model(coords=coords) as logistic_regression_model:
        x = pm.ConstantData("doses", data['doses'].values, dims="observation")
        # priors
        beta0 = pm.Normal("beta0", mu=-3, sigma=2) # Made up by myself
        beta1 = pm.Exponential("beta1", lam=1) # given from the paper
        # linear model
        mu = beta0 + beta1 * x
        # probabilities
        p = pm.Deterministic("p", sigmoid(mu), dims="observation")
        # likelihood
        y_obs = pm.Bernoulli("y", logit_p=mu, observed=data['toxicity_event'].values, dims="observation") 
    return logistic_regression_model
    
    
def get_next_dose(idata):
    # Extract betas from the trace
    beta0 = np.mean(idata.posterior['beta0'].values)
    beta1 = np.mean(idata.posterior['beta1'].values)
    # store the doses in a numpy array
    x_values = np.array(doses)
    # Generate y values (sigmoid probabilities)
    y_values = sigmoid(beta0 + beta1 * x_values)
    # select the highest dose that falls under the MTD threshold (0.33)
    # if the lowest dose is predicted to be above the threshold, return the next dose INDEX equal to 0 which is a dose of 0.5
    if len(y_values[y_values <= 0.33]) == 0:
        next_dose = 0
    else:
        next_dose = np.argmax(y_values[y_values <= 0.33])
    return next_dose


def run_single_trial():
    """ The setup is as follows:
    1. Choose a starting dose to test on the first three patients. 
    2. Simulate whether each patient has a toxicity event with sim_data().
    3. run the model with the data collected so far. 
    4. Using the posterior estimates of the beta's, predict the highest dose that has a predicted probability of toxicity below 0.33.
    5. Repeat steps 2-4 until all 36 patients have been tested.
    
    numpyro appears to be a faster engine and is able to run the model in a reasonable amount of time. There are other options 
    such as variational inference or even quadratic approximation. However, I am not familiar with these methods.
    """
    # first run of the model with a starting dose of 1 (index of 1)
    data = sim_data(0) 
    logistic_regression_model = run_model(data)
    with logistic_regression_model:
        idata = pm.sample(500, tune=200, chains=2, random_seed=rng, cores=4, progressbar=False, nuts_sampler='numpyro')
    next_dose = get_next_dose(idata)
    
    # We already went through 3 samples out of 36. 36 // 3 = 12 - 1 = 11
    for remaining_trials in range(11):
        new_data = sim_data(next_dose)
        # combine the data collected previously with the new simulated data
        data = pd.concat([data, new_data], axis=0, ignore_index=True)
        # run the model on all the data collected so far
        logistic_regression_model = run_model(data)
        with logistic_regression_model:
            idata = pm.sample(500, tune=200, chains=2, random_seed=rng, cores=4, progressbar=False, nuts_sampler='numpyro')
        next_dose = get_next_dose(idata)
    
    beta0 = np.mean(idata.posterior['beta0'].values)
    beta1 = np.mean(idata.posterior['beta1'].values)
    # store the doses in a numpy array
    x_values = np.array(doses)
    # Generate y values (sigmoid probabilities)
    y_values = sigmoid(beta0 + beta1 * x_values)
    return data, y_values


total_data = []
sigmoid_probabilities = []
# we should have 1000 simulations in total
for sim in tqdm(range(700)):
    data, trial_probabilities = run_single_trial()
    total_data.append(data)    
    sigmoid_probabilities.append(trial_probabilities)
    
# save the sigmoid probabilities to a pickle file
with open("sigmoid_probs_pt5.pkl", "wb") as f: # "wb" because we want to write in binary mode
    pickle.dump(sigmoid_probabilities, f)
    
    
for i in range(len(total_data)):
    # add a column that tracks which simulation run is associated with each row
    total_data[i]['sim_run'] = i
    
# combine list of dataframes to one dataframe
total_data = pd.concat(total_data) 

# update a new file name based on the files already available
rel_files = [file.startswith("finished_simulations_") for file in os.listdir("gabe_sim_files")]
new_index = int(max("finished_simulations_pt1.parquet"[23])) + 1
new_file_name = f"finished_simulations_pt{str(new_index)}.parquet"
total_data.to_parquet(f'gabe_sim_files/{new_file_name}')