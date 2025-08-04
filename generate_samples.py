import pandas as pd
from utils.util_functions import *
from sampler_functions.sampler import *
from sampler_functions.lower import *
from sampler_functions.upper import *
from sampler_functions.midpoint import *
from sampler_functions.random import *

"""
Generating samples of intervals from LCNs 
"""

# Define a list of LCNs to create interval datasets for
lcns = ["psych_small", "psych_medium"]

# def generate_interval_data(dataset_name, filepath):
#     lcn_data = load_json_data(filepath)
#     print(lcn_data)

#     #Generate a dataset of 100 rows
#     samples = generate_dataset(lcn_data, n=100)

#     # Convert to DataFrame if needed
#     df = pd.DataFrame(samples)
    
#     # Save to csv
#     save_sampled_intervals(df, dataset_name)

#     return df


def generate_all_method_samples(dataset_name, filepath):
    lcn_data = load_json_data(filepath)
    print(lcn_data)

    # Generate datasets of 100 rows

    # Lower probability
    lower_df = lower_probability_sampler(lcn_data)
    # Save to csv
    save_sampled_intervals(lower_df, 'lower', dataset_name)


    # Upper Probability 
    upper_df = upper_probability_sampler(lcn_data)
    save_sampled_intervals(upper_df, 'upper', dataset_name)


    # Midpoint Probability 
    midpoint_df = midpoint_probability_sampler(lcn_data)
    save_sampled_intervals(midpoint_df, 'midpoint', dataset_name)


    # Random Probability 
    random_df = random_probability_sampler(lcn_data)
    save_sampled_intervals(random_df, 'random', dataset_name)


    return 'Created datasets for all sampling methods'



def run_sampling_workflow():

    for lcn in lcns:
        filepath = f"datasets/lcns/{lcn}.json"

        generate_all_method_samples(lcn, filepath)
    
    print("Completed Sampling")


run_sampling_workflow()