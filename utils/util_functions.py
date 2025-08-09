import json
import pandas as pd

def load_json_data(filename): 
    # Open and read the JSON file
    with open(filename, 'r') as file:
        data = json.load(file)

    # return the data
    return data


def save_json_data(filename, data):
    # Save json data according to specified name
    json_str = json.dumps(data, indent=4)
    with open(f"{filename}.json", "w") as f:
        f.write(json_str)


def save_outputs(results, dataset_name):
    # Save text of results
    with open(f"outputs/{dataset_name}_results.txt", "w") as f:
        f.write(results)


def save_sampled_intervals(dataframe, method, lcn_name):
    # Saves sampled dataset as csv 
    filename = f"{lcn_name}_{method}_samples.csv"
    dataframe.to_csv(f"datasets/sampled_data/{filename}", index=False)


def load_csv_data(filename):
    # Loads data from csv and returns dataframe
    df = pd.read_csv(f"datasets/sampled_data/{filename}.csv")

    return df