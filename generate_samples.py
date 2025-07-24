import pandas as pd
from utils.util_functions import *
from lcn_functions.sampler import *

"""
Generating samples of intervals from LCNs 
"""

# Define a list of LCNs to create interval datasets for
lcns = ["net1"]

def generate_interval_data(dataset_name, filepath):
    lcn_data = load_json_data(filepath)
    print(lcn_data)

    #Generate a dataset of 100 rows
    samples = generate_dataset(lcn_data, n=100)

    # Convert to DataFrame if needed
    df = pd.DataFrame(samples)
    
    # Save to csv
    save_sampled_intervals(df, dataset_name)

    return df


def run_sampling_workflow():

    for lcn in lcns:
        filepath = f"datasets/lcns/{lcn}.json"

        generate_interval_data(lcn, filepath)
    
    print("Completed Sampling")


run_sampling_workflow()