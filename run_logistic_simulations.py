import numpy as np
import pandas as pd
import pymc as pm
import warnings 
from pymc.sampling import jax

RANDOM_SEED = 58
rng = np.random.default_rng(RANDOM_SEED)
warnings.filterwarnings("ignore")

doses = np.array([0.5, 1, 3, 5, 6]) # From figure 6 and in units (mg/m^2 per day)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def sim_data(relevant_index, true_toxic_prob):
    """ Simulates whether any of the 3 participants have a toxicity event given by the unknown probabilities.

    Args:
        relevant_index (Int): The index associated with the dose being used on the three samples.

    Returns:
        Pandas Dataframe: Contains the doses the three participants took and whether each participant had a toxicity event
    """
    toxicity_observation = rng.binomial(1, p=true_toxic_prob[relevant_index], size=3) # 3 samples per cohort
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
    with pm.Model() as logistic_regression_model:
        x = pm.ConstantData("doses", data['doses'].values)
        # priors
        beta0 = pm.Normal("beta0", mu=-3, sigma=1) # Made up by myself
        beta1 = pm.Exponential("beta1", lam=2) # given from the paper
        # linear model
        mu = beta0 + beta1 * x
        # probabilities
        # p = pm.Deterministic("p", pm.math.sigmoid(mu))
        # likelihood
        # y_obs = pm.Bernoulli("y", logit_p=pm.math.sigmoid(mu), observed=data['toxicity_event'].values) 
        y_obs = pm.Logistic("y", mu=mu, observed=data['toxicity_event'].values) 
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
        idata = jax.sample_numpyro_nuts(850, 350, cores=5, target_accept=0.70, progressbar=False)
    next_dose = get_next_dose(idata)
    
    # We already went through 3 samples out of 36. 36 // 3 = 12 - 1 = 11
    for remaining_trials in range(11):
        new_data = sim_data(next_dose)
        # combine the data collected previously with the new simulated data
        data = pd.concat([data, new_data], axis=0, ignore_index=True)
        # run the model on all the data collected so far
        logistic_regression_model = run_model(data)
        with logistic_regression_model:
            idata = jax.sample_numpyro_nuts(850, 350, cores=5, target_accept=0.70, progressbar=False)
        next_dose = get_next_dose(idata)
    
    beta0 = np.mean(idata.posterior['beta0'].values)
    beta1 = np.mean(idata.posterior['beta1'].values)
    # Generate y values (sigmoid probabilities)
    y_values = sigmoid(beta0 + beta1 * doses)
    return data, y_values