from lcn_functions.model import create_lcn
from metric_functions.kl_divergence import kl_divergence_from_samples
from metric_functions.structural_hamming_distance import structural_hamming_distance_compare
from sampler_functions.bn_topological import ancestral_sample_bn, build_precise_bn_from_lcn
from sampler_functions.contingency_sampler import credal_aggregate_intervals, sample_dataset
import pandas as pd
from scoring_functions.interval_bic_derivation import compute_interval_BIC, compute_network_interval_BIC
from structure_learning.hill_climbing import run_hillclimbing_bic, run_interval_bic_hillclimb
from structure_learning.sim_anneal import run_simanneal_sa
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator


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

        model, sampled_states = build_precise_bn_from_lcn(lcn)

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
    print("Done 1")

    # Run Simulated Annealing
    #sa_edges, sa_score = run_simanneal_sa(bn_samples)
    print("Done 2")

    learned_bn = {
        "hillclimb_edges": hc_edges,
        "hillclimb_bic": hc_score,
    }

    return learned_bn


def run_structural_hamming_distance(true_model, learned_bn_dict):
    """
    Compute SHD between the true baseline BN and the learned baseline BN
    (both HillClimbing and Simulated Annealing).
    """

    print(learned_bn_dict)

    true_edges = list(true_model.edges())

    hc_edges = learned_bn_dict["hillclimb_edges"]

    shd_hc = structural_hamming_distance_compare(true_edges, hc_edges)
    print(shd_hc)


    return {
        "hillclimb_shd": shd_hc
    }


def interval_bic_structure_learn(credal_aggregate_table, lcn_forward_samples):
    """
    Run greedy hill climbing with interval BIC, returning three networks
    for lower, mid, and upper BIC.
    """
    results = {}
    for scoring in ['low', 'mid', 'high']:
        edges, score = run_interval_bic_hillclimb(lcn_forward_samples, credal_aggregate_table, scoring)
        results[scoring] = {'edges': edges, 'score': score}

    # Compute the full network interval BIC from the contingency table
    interval_per_node = compute_interval_BIC(credal_aggregate_table)
    network_interval = compute_network_interval_BIC(interval_per_node)

    results['network_interval'] = network_interval  # [lower, upper]
    
    return results


def run_interval_lcn_shd(true_model, interval_bic_results):
    """
    Compute SHD between the true baseline BN structure and the
    LCN structures learned using interval BIC (low, mid, high).
    """
    
    true_edges = list(true_model.edges())
    
    shd_results = {}
    
    for scoring in ['low', 'mid', 'high']:
        learned_edges = list(interval_bic_results[scoring]['edges'])
        
        shd_value = structural_hamming_distance_compare(true_edges, learned_edges)
        
        shd_results[f"{scoring}_shd"] = shd_value

    return shd_results


def build_bn_from_lcn_learned(interval_bic_results, baseline_bn, baseline_bn_samples):
    # Since all 3 LCN-BNs from 'Interval BIC Results' have the same structure, just need to evaluate one (mid) 
    
    # Extract the learned BIC structure 
    lcn_edges = list(interval_bic_results['mid']['edges'])

    # Convert to a list explicitly for pgmpy
    lcn_edges = list(lcn_edges)

     # Build a BN from the LCN-learned structure
    lcn_learned_bn = DiscreteBayesianNetwork(lcn_edges)

    # Ensure node alignment with the baseline BN
    lcn_learned_bn.add_nodes_from(baseline_bn.nodes())

    # Fit Paramerters using baseline BN forward samples
    lcn_learned_bn.fit(
        baseline_bn_samples,
        estimator=MaximumLikelihoodEstimator
    )

    # Now, there is a fully specified BN, Structure came from Interval-BIC, Parameters came from Data
    return lcn_learned_bn



def run_workflow_config(size, interval_width, width_dist_type, in_degree, num_samples):
    """
    Run each step of the workflow
    """

    # (1) Generate LCN
    gen_lcn = generate_lcn_with_params(size, interval_width, width_dist_type, in_degree)


    # (2.1) Sample Baseline Bayesian Network
    model, sampled_states = generate_baseline_bn(gen_lcn)
    print('Model')
    print(model)

    
    # (2.2) Sample LCN and create Contingency Table
    lcn_aggregate_table, lcn_samples_df = contingency_sample_lcn(gen_lcn, num_samples)


    # (3.1) Ancestral Sampling Baseline BN
    bn_forward_samples = ancestral_sample_bn(model, n_samples=num_samples)


    # (3.2) Learn Baseline BN structure using standard BIC and heuristic approaches
    learned_bn = structure_learn_baseline_bn(bn_forward_samples)
    print(learned_bn)


    # (3.3) Compute SHD between baseline and learned BN
    baseline_shd_results = run_structural_hamming_distance(model, learned_bn)


    # (4.1) Learn LCN structure with interval BIC and heuristic approaches
    interval_bic_results = interval_bic_structure_learn(lcn_aggregate_table, lcn_samples_df)
    print('Interval BIC Results')
    print(interval_bic_results)


    # (4.2) Compute SHD between baseline BN and learned LCN structures 
    interval_lcn_shd_results = run_interval_lcn_shd(model, interval_bic_results)
    print(interval_lcn_shd_results)


    # (5) Compute KL divergence between distributions 

    # (5.1) KL divergence between baseline BN and Learned BN 

    print("Params")
    print("model", model)
    print("learned_bn", learned_bn)
    print("forward samples", bn_forward_samples)

    # Extract structure
    edges = list(learned_bn['hillclimb_edges']) 

    # Build BN
    learned_bn_model = DiscreteBayesianNetwork(edges)
    learned_bn_model.add_nodes_from(model.nodes())

    # Fit parameters
    learned_bn_model.fit(bn_forward_samples)

    kl_baseline_vs_learned = kl_divergence_from_samples(
        true_model=model,
        approx_model=learned_bn_model,
        samples_df=bn_forward_samples
    )

    print("KL Baseline vs Learned BN")
    print(kl_baseline_vs_learned)


    # (5.2) Compute KL Divergence between baseline BN and LCN-Learned BN 

    # Get specified LCN-Learned BN to use 
    lcn_learned_bn = build_bn_from_lcn_learned(interval_bic_results, model, bn_forward_samples)

    # Compute KL Divergence
    kl_baseline_vs_lcn_learned = kl_divergence_from_samples(
        true_model=model,
        approx_model=lcn_learned_bn,
        samples_df=bn_forward_samples
    )

    print("KL Baseline vs LCN-learned")
    print(kl_baseline_vs_lcn_learned)


    # (6) Return results and data for saving
    
    return {
        "config": {
            "size": size,
            "interval_width": interval_width,
            "width_dist_type": width_dist_type,
            "in_degree": in_degree,
            "num_samples": num_samples,
        },
        "lcn": gen_lcn,
        "baseline_bn": {
            "model": model,
            "sampled_states": sampled_states,
            "forward_samples": bn_forward_samples,
        },
        "baseline_structure_learning": {
            "learned_bn": learned_bn,
            "shd": baseline_shd_results,
        },
        "interval_bic_learning": {
            "interval_bic_results": interval_bic_results,
            "interval_lcn_shd": interval_lcn_shd_results,
        },
        "kl_divergence": {
            "baseline_vs_learned_bn": kl_baseline_vs_learned,
            "baseline_vs_lcn_learned_bn": kl_baseline_vs_lcn_learned,
        },
        "intermediate": {
            "lcn_aggregate_table": lcn_aggregate_table,
            "lcn_samples_df": lcn_samples_df,
            "learned_bn_model": learned_bn_model,
            "lcn_learned_bn_model": lcn_learned_bn,
        }
    }




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

    # Parameters 
    size = 5
    interval_width = 0.2
    width_dist_type = "beta"
    in_degree = 1
    num_samples = 300


    results1 = run_workflow_config(size, interval_width, width_dist_type, in_degree, num_samples)

    # Returning results and params
    experiment_obj = {
        "params": {
            "size": size,
            "interval_width": interval_width,
            "width_dist_type": width_dist_type,
            "in_degree": in_degree,
            "num_samples": num_samples,
        },
        "results": results1
    }

    return experiment_obj