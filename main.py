import pandas as pd
from utils.util_functions import load_json_data
from lcn_functions.sampler import *
from structure_learning.hill_climbing import *


def generate_sample_dataset(dataset_name):
    lcn_data = load_json_data(dataset_name)
    print(lcn_data)

    #Generate a dataset of 100 rows
    samples = generate_dataset(lcn_data, n=100)

    # Convert to DataFrame if needed
    df = pd.DataFrame(samples)
    df.to_csv('lcn_samples.csv', index=False)

    print(df)

    struc = run_hillclimb_search(df)

    print(struc)

generate_sample_dataset("datasets/net1.json")