import pandas as pd
from utils.util_functions import *
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


def learn_structure(dataset_name):

    results_output = ""

    # Generate sample dataset
    dataset = load_csv_data(dataset_name)

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
    save_outputs(results_output, dataset_name)

    print("Completed structure learning and saved outputs")


def run_workflow():
    learn_structure('psych_small_lower_samples')

    learn_structure('psych_small_upper_samples')

    learn_structure('psych_small_midpoint_samples')

    learn_structure('psych_small_random_samples')

# Run main method
#run_workflow()


def learn_structure_lcn_samples(dataset_name, dataset):
    """
    Use the forward sampled LCN data to structure learn. 
    - Greedy Hill Climbing
    - Simulated Annealing 
    
    Returns:
        dict with arrays of edges for each method
    """

    results_output = ""

    # Hill Climbing structure learning
    hc_edges = run_hillclimb_search(dataset)

    # Create and save output
    results1 = structure_results("Hill Climbing", hc_edges)
    results_output += results1

    # Simulated Annealing Search
    sa_edges = simulated_annealing_search(dataset)
    results2 = structure_results("Simulated Annealing", sa_edges)
    results_output += results2

    # Save final results
    #save_outputs(results_output, dataset_name)

    print("Completed structure learning and saved outputs")

    # Return edges in arrays
    result = {
        "hill_climbing": list(hc_edges),
        "sim_annealing": list(sa_edges)
    }

    return result