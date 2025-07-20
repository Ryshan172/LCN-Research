import pandas as pd
from utils.util_functions import *
from lcn_functions.sampler import *
from structure_learning.hill_climbing import *
from structure_learning.simulated_annealing import simulated_annealing_search

def structure_results(method, results):
    result_summary = ""
    result_summary += f"Structure Learning Method: {method}\n"
    result_summary += "Learned Edges:\n"

    # Convert results (likely an OutEdgeView) into a list of edges
    for edge in list(results):
        result_summary += f"  {edge[0]} -> {edge[1]}\n"

    return result_summary


def generate_sample_dataset(dataset_name):
    lcn_data = load_json_data(dataset_name)
    print(lcn_data)

    #Generate a dataset of 100 rows
    samples = generate_dataset(lcn_data, n=100)

    # Convert to DataFrame if needed
    df = pd.DataFrame(samples)
    df.to_csv('lcn_samples.csv', index=False)

    return df


def learn_structure_workflow():

    results_output = ""

    # Generate sample dataset
    dataset = generate_sample_dataset("datasets/net1.json")

    # Hill Climbing structure learning
    struc = run_hillclimb_search(dataset)

    # Create and save output
    results1 = structure_results("Hill Climbing", struc)
    results_output += results1


    # Simulated Annealing Search
    struc_sa = simulated_annealing_search(dataset)
    results2 = structure_results("Simulated Annealing", struc_sa)
    results_output += results2

    # Save final results
    save_outputs(results_output)

    print("Completed structure learning and saved outputs")


# Run main method
learn_structure_workflow()