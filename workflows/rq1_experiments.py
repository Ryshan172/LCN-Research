from lcn_functions.model import create_lcn
from metric_functions.structural_hamming_distance import structural_hamming_distance_compare
from sampler_functions.bn_topological import ancestral_sample_bn, build_precise_bn_from_lcn
from sampler_functions.contingency_sampler import credal_aggregate_intervals, sample_dataset
import pandas as pd
from structure_learning.hill_climbing import run_hillclimbing_bic
from structure_learning.sim_anneal import run_simanneal_sa


"""
Research Question 1 Experiments Workflow
"""


def generate_lcn_with_params(size, interval_width, width_dist_type, in_degree):
    """
    Generate an LCN for the given parameters
    """

    lcn = create_lcn(size, interval_width, width_dist_type, in_degree)

    return lcn



def generate_baseline_bn(lcn):
    """
    Keeps sampling a precise BN from an LCN until the model is valid.
    Avoids infinite loops by enforcing a retry limit.
    """

    max_retries = 20

    for attempt in range(1, max_retries + 1):
        print(f"\nAttempt {attempt}/{max_retries}")

        lcn_dict = lcn.dict(by_alias=True)
        model, sampled_states = build_precise_bn_from_lcn(lcn_dict)

        # Sanity check
        model_correct = model.check_model()
        print(f"Model correct: {model_correct}")

        if model_correct:
            print("Valid model generated.")
            return model, sampled_states

    # If loop ends with no valid model
    raise RuntimeError(
        f"Failed to generate a valid BN after {max_retries} attempts."
    )



def contingency_sample_lcn(lcn, num_samples):
    """
    Run aggregate sampler functions on lcn 
    
    returns:
    - Aggregate table as dataframe
    - Ancestral samples dataset of LCN
    """

    # Run topological sampling of LCN
    forward_samples = sample_dataset(lcn, num_samples)
    
    # Convert to dataframe
    lcn_samples_df = pd.DataFrame(forward_samples)

    # Compute aggregate contingency table
    lcn_aggregate_table = credal_aggregate_intervals(forward_samples, lcn)

    return lcn_aggregate_table, lcn_samples_df



def structure_learn_baseline_bn(bn_samples):
    """
    Learn the Baseline Bayesian Network using the standard BIC score and heuristic approaches
    """

    #  Run Greedy Hill Climbing 
    hc_edges, hc_score = run_hillclimbing_bic(bn_samples)

    # Run Simulated Annealing
    sa_edges, sa_score = run_simanneal_sa(bn_samples)

    learned_bn = {
        "hillclimb_edges": hc_edges,
        "hillclimb_bic": hc_score,
        "simanneal_edges": sa_edges,
        "simanneal_bic": sa_score
    }

    return learned_bn


def run_structural_hamming_distance(true_model, learned_bn_dict):
    """
    Compute SHD between the true baseline BN and the learned baseline BN
    (both HillClimbing and Simulated Annealing).
    """

    true_edges = list(true_model.edges())

    hc_edges = learned_bn_dict["hillclimb_edges"]
    sa_edges = learned_bn_dict["simanneal_edges"]

    shd_hc = structural_hamming_distance_compare(true_edges, hc_edges)
    shd_sa = structural_hamming_distance_compare(true_edges, sa_edges)

    return {
        "hillclimb_shd": shd_hc,
        "simanneal_shd": shd_sa
    }



def run_workflow_config(size, interval_width, width_dist_type, in_degree, num_samples):
    """
    Run each step of the workflow
    """

    # (1) Generate LCN
    gen_lcn = generate_lcn_with_params(size, interval_width, width_dist_type, in_degree)


    # (2.1) Sample Baseline Bayesian Network
    model, sampled_states = generate_baseline_bn(gen_lcn)

    
    # (2.2) Sample LCN and create Contingency Table
    lcn_aggregate_table, lcn_samples_df = contingency_sample_lcn(gen_lcn, num_samples)


    # (3.1) Ancestral Sampling Baseline BN
    bn_forward_samples = ancestral_sample_bn(model, n_samples=num_samples)


    # (3.2) Learn Baseline BN structure using standard BIC and heuristic approaches
    learned_bn = structure_learn_baseline_bn(bn_forward_samples)


    # (3.3) Compute SHD between baseline and learned BN
    shd_results = run_structural_hamming_distance(model, learned_bn)


    # (4.1) Learn LCN structure with interval BIC and heuristic approaches


    # (4.2) Compute SHD between baseline BN and learned structure 


    # (5) Compute KL divergence between distributions 


    return 



def experiment_run_controller():
    """
    Setup and control variations for experiments 
    Saving results and variables used 

    Run variations of workflow by changing: 

    - 1. LCN size
    - 2. Interval Width
    - 3. Width distribution 
    - 4. In-degree
    - 5. Number of samples in ancestral sampling (100, 200, 300 etc)
    """

    return