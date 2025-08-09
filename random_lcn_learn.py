import pandas as pd
from utils.util_functions import *
from sampler_functions.sampler import *
from structure_learning.hill_climbing import *
from structure_learning.simulated_annealing import simulated_annealing_search
from utils.util_functions import *
from sampler_functions.lower import *
from sampler_functions.upper import *
from sampler_functions.midpoint import *
from sampler_functions.random import *
from lcn_functions.model import generate_lcn_workflow


def structure_results(method, results):
    result_summary = ""
    result_summary += f"Structure Learning Method: {method}\n"
    result_summary += "Learned Edges:\n"

    # Convert results (likely an OutEdgeView) into a list of edges
    for edge in list(results):
        result_summary += f"  {edge[0]} -> {edge[1]}\n"

    return result_summary


def learn_structure(dataset_name, dataset):
    """
    Uses input dataset and learns structure.
    Saves output using dataset name
    """

    results_output = ""


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

    print(f"Completed structure learning for {dataset_name} and saved outputs")



def generate_samples_and_learn(dataset_name, lcn_data):

    # Generate datasets of 100 rows

    # Lower probability sampling and structure learning
    lower_df = lower_probability_sampler(lcn_data)
    learn_structure("random_lower_df", lower_df)

    """
    # Upper Probability 
    upper_df = upper_probability_sampler(lcn_data)
    save_sampled_intervals(upper_df, 'upper', dataset_name)


    # Midpoint Probability 
    midpoint_df = midpoint_probability_sampler(lcn_data)
    save_sampled_intervals(midpoint_df, 'midpoint', dataset_name)


    # Random Probability 
    random_df = random_probability_sampler(lcn_data)
    save_sampled_intervals(random_df, 'random', dataset_name)
    """


    return f"Created all sample dataset for {dataset_name}"



def run_workflow():

    # Generate a random LCN
    lcn_data = generate_lcn_workflow()

    generate_samples_and_learn("test_lcn", lcn_data)



# Run main method
run_workflow()