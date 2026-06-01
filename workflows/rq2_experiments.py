
import pandas as pd
import itertools

from lcn_functions.model import create_lcn
from sampler_functions.contingency_sampler import sample_dataset, credal_aggregate_intervals
from metric_functions.structural_hamming_distance import structural_hamming_distance_compare
from metric_functions.kl_divergence import kl_divergence_from_samples
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator


"""
Experiments workflow for Research Question 2
"""

#-------------- LCN Generation and Sampling ----------------------------
def generate_lcn(size, interval_width, width_dist_type, in_degree):
    return create_lcn(size, interval_width, width_dist_type, in_degree)


def sample_lcn(lcn, num_samples):
    """
    Returns:
    - samples_df
    - aggregate_table
    """

    samples = sample_dataset(lcn, num_samples)
    samples_df = pd.DataFrame(samples)

    aggregate_table = credal_aggregate_intervals(samples, lcn)

    return samples_df, aggregate_table


#-------------- Candidate LCN graphs --------------------------
def initialise_candidate_population(nodes, population_size=20):
    """
    TODO:
    - random DAGs
    - LCN-informed perturbations
    - sparse adjacency matrices
    """
    pass


# ---------------- Mutations and Heuristic Optimisation Strategies ------------------
def apply_mutation_strategy(population, mutation_type="edge_flip"):
    """
    TODO:
    Mutation operators:
    - edge_flip
    - edge_add
    - edge_delete
    - interval-aware mutation
    """
    pass

# Interval BIC based
def run_heuristic_search(samples_df, aggregate_table, method="hill_climbing", scoring="mid"):
    """
    TODO:
    Plug-in optimizers:
    - hill climbing
    - simulated annealing
    - tabu search

    Returns:
    - best_edges
    - score_trajectory
    """
    pass


#-------------------- Evaluate Interval BIC Score --------------------------

def evaluate_interval_bic(score_trajectory):
    """
    TODO:
    - best score
    - convergence speed
    - stability (variance over time)
    """
    return {
        "best_score": None,
        "convergence_speed": None,
        "stability": None
    }


#--------------------- Metrics --------------------------------------

# SHD
def compute_shd(true_model, learned_edges):
    return structural_hamming_distance_compare(
        list(true_model.edges()),
        list(learned_edges)
    )

# F1, Accuracy, Precision, Recall
def compute_edge_metrics(true_model, learned_edges, nodes):
    """
    Edge-level classification metrics:
    Accuracy, Precision, Recall, F1
    """

    true_edges = set(true_model.edges())
    learned_edges = set(learned_edges)

    all_possible = {
        (i, j)
        for i in nodes
        for j in nodes
        if i != j
    }

    tp = len(true_edges & learned_edges)
    fp = len(learned_edges - true_edges)
    fn = len(true_edges - learned_edges)
    tn = len(all_possible - (true_edges | learned_edges))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    f1 = (2 * precision * recall) / (precision + recall + 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1
    }

# KL Divergence
def compute_kl(true_model, learned_model, samples_df):
    """
    Distributional comparison:
    KL(true || learned)
    """

    return kl_divergence_from_samples(
        true_model=true_model,
        approx_model=learned_model,
        samples_df=samples_df
    )


def run_experiment_steps(size, interval_width, width_dist_type, in_degree, num_samples):

    # (1) LCN
    lcn = generate_lcn(size, interval_width, width_dist_type, in_degree)

    # (2) Sampling
    samples_df, aggregate_table = sample_lcn(lcn, num_samples)

    # (3) Candidate population
    population = initialise_candidate_population(lcn.nodes())

    # (4) Mutation
    mutated_population = apply_mutation_strategy(population)

    # (5) Heuristic search
    best_edges, trajectory = run_heuristic_search(
        samples_df,
        aggregate_table,
        method="hill_climbing",
        scoring="mid"
    )

    # (6) Interval BIC performance
    bic_metrics = evaluate_interval_bic(trajectory)

    # (7a) SHD
    shd = compute_shd(lcn, best_edges)

    # (7b) Edge metrics
    edge_metrics = compute_edge_metrics(
        lcn,
        best_edges,
        lcn.nodes()
    )

    # (7c) KL divergence (optional but important)
    learned_bn = DiscreteBayesianNetwork(best_edges)
    learned_bn.add_nodes_from(lcn.nodes())
    learned_bn.fit(samples_df, estimator=MaximumLikelihoodEstimator)

    kl = compute_kl(lcn, learned_bn, samples_df)

    # Return full structured results
    return {
        "config": {
            "size": size,
            "interval_width": interval_width,
            "width_dist_type": width_dist_type,
            "in_degree": in_degree,
            "num_samples": num_samples,
        },
        "data": {
            "samples_df": samples_df,
            "aggregate_table": aggregate_table
        },
        "structure": {
            "learned_edges": best_edges,
            "shd": shd
        },
        "edge_metrics": edge_metrics,
        "kl_divergence": kl,
        "interval_bic": bic_metrics,
        "trajectory": trajectory
    }