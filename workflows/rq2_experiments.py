import pandas as pd
import itertools
import numpy as np
import copy

from lcn_functions.model import create_lcn
from sampler_functions.contingency_sampler import sample_dataset, credal_aggregate_intervals
from metric_functions.structural_hamming_distance import structural_hamming_distance_compare
from metric_functions.kl_divergence import kl_divergence_from_samples
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator


"""
RQ2: Heuristic Structure Learning over LCNs using Interval BIC
- mutation strategies
- hill climbing / random restart / tabu search
- constraint-aware structure learning
"""


# ----------- 1. LCN GENERATION ------------------------------------
def generate_lcn(size, interval_width, width_dist_type, in_degree):
    return create_lcn(size, interval_width, width_dist_type, in_degree)


def sample_lcn(lcn, num_samples):
    samples = sample_dataset(lcn, num_samples)
    samples_df = pd.DataFrame(samples)

    aggregate_table = credal_aggregate_intervals(samples, lcn)

    return samples_df, aggregate_table


# -------- 2. INITIAL STRUCTURE -------------------------
def initialise_structure(nodes, strategy="empty"):
    """
    Instead of returning an edge list, 
    returns FULL LCN STATE (edges + constraints + credal sets placeholder)

    WHY:
    - mutation operates on LCN structure, not just edges
    - constraints must persist through search
    """

    return {
        "nodes": list(nodes),
        "edges": [],
        "credal_sets": {},
        "logical_constraints": []
    }


# ----- 3. MUTATION OPERATORS ------------
def mutate_graph(lcn_state, mutation_type="edge_add", max_attempts=10):
    """
    Mutates full LCN state instead of just edges 

    Why:
    - constraints must be checked during mutation
    - credal structure must stay consistent with edges
    """

    nodes = lcn_state["nodes"]
    edges = copy.deepcopy(lcn_state["edges"])

    for _ in range(max_attempts):

        candidate = copy.deepcopy(lcn_state)
        candidate_edges = copy.deepcopy(edges)

        if mutation_type == "edge_add":
            a, b = np.random.choice(nodes, 2, replace=False)

            if (a, b) not in candidate_edges:
                candidate_edges.append((a, b))

        elif mutation_type == "edge_delete":
            if candidate_edges:
                candidate_edges.pop(np.random.randint(len(candidate_edges)))

        elif mutation_type == "edge_flip":
            if candidate_edges:
                i = np.random.randint(len(candidate_edges))
                a, b = candidate_edges[i]
                candidate_edges[i] = (b, a)

        # update candidate state
        candidate["edges"] = candidate_edges

        return candidate

    return lcn_state


# -------- 4. SCORE FUNCTION ------------------
def score_structure(lcn_state, samples_df, aggregate_table, scoring="mid"):
    """
    Scores full LCN state instead of edges only

    Why:
    - interval BIC depends on structure + credal interpretation
    - future constraint-aware scoring will require full LCN
    """

    edges = lcn_state["edges"]

    # TODO: replace with Interval BIC implementation
    return 0.0


# -------- 5. HILL CLIMBING CORE -------------------
def hill_climb(
    samples_df,
    aggregate_table,
    nodes,
    mutation_type="edge_add",
    max_iter=100
):

    # Initialises FULL LCN state
    current = initialise_structure(nodes)
    current_score = score_structure(current, samples_df, aggregate_table)

    best = copy.deepcopy(current)
    best_score = current_score

    trajectory = [current_score]

    for _ in range(max_iter):

        # Mutation operates on full LCN state
        candidate = mutate_graph(current, mutation_type)
        candidate_score = score_structure(candidate, samples_df, aggregate_table)

        if candidate_score > current_score:
            current = candidate
            current_score = candidate_score

            if candidate_score > best_score:
                best = copy.deepcopy(candidate)
                best_score = candidate_score

        trajectory.append(current_score)

    return best, trajectory


# ---------- 6. RANDOM RESTART ------------------------
def random_restart_hill_climb(
    samples_df,
    aggregate_table,
    nodes,
    mutation_type="edge_add",
    n_restarts=10,
    max_iter=50
):

    best_overall = None
    best_score = -np.inf
    full_trajectory = []

    for _ in range(n_restarts):

        result, traj = hill_climb(
            samples_df,
            aggregate_table,
            nodes,
            mutation_type,
            max_iter
        )

        final_score = traj[-1]
        full_trajectory.extend(traj)

        if final_score > best_score:
            best_score = final_score
            best_overall = result

    return best_overall, full_trajectory


# ------------- 7. TABU SEARCH ------------------
def tabu_search(
    samples_df,
    aggregate_table,
    nodes,
    mutation_type="edge_add",
    tabu_size=10,
    max_iter=100
):

    current = initialise_structure(nodes)
    current_score = score_structure(current, samples_df, aggregate_table)

    best = copy.deepcopy(current)
    best_score = current_score

    tabu_list = []
    trajectory = [current_score]

    for _ in range(max_iter):

        candidate = mutate_graph(current, mutation_type)

        # Tabu must compare LCN STATES, not raw edges
        if any(candidate["edges"] == t["edges"] for t in tabu_list):
            continue

        candidate_score = score_structure(candidate, samples_df, aggregate_table)

        tabu_list.append(copy.deepcopy(candidate))

        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        if candidate_score > current_score:
            current = candidate
            current_score = candidate_score

        if candidate_score > best_score:
            best = copy.deepcopy(candidate)
            best_score = candidate_score

        trajectory.append(current_score)

    return best, trajectory


# ------------ 8. EVALUATION METRICS ------------------
def evaluate_interval_bic(trajectory):
    return {
        "best_score": max(trajectory),
        "convergence_speed": len(trajectory),
        "stability": np.var(trajectory)
    }


def compute_shd(true_model, learned_edges):
    return structural_hamming_distance_compare(
        list(true_model.edges()),
        list(learned_edges)
    )


def compute_edge_metrics(true_model, learned_edges, nodes):

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


def compute_kl(true_model, learned_model, samples_df):
    return kl_divergence_from_samples(
        true_model=true_model,
        approx_model=learned_model,
        samples_df=samples_df
    )


# ---------- 9. EXPERIMENT RUNNER -------------
def run_experiment_steps(
    size,
    interval_width,
    width_dist_type,
    in_degree,
    num_samples,
    search_method="hill_climbing",
    mutation_type="edge_add"
):

    # (1) Ground-truth LCN
    lcn = generate_lcn(size, interval_width, width_dist_type, in_degree)

    # (2) Sampling
    samples_df, aggregate_table = sample_lcn(lcn, num_samples)

    nodes = list(lcn.nodes())

    # Receives FULL LCN as output, not edges only
    if search_method == "hill_climbing":

        best_lcn, trajectory = hill_climb(
            samples_df,
            aggregate_table,
            nodes,
            mutation_type
        )

    elif search_method == "random_restart":

        best_lcn, trajectory = random_restart_hill_climb(
            samples_df,
            aggregate_table,
            nodes,
            mutation_type
        )

    elif search_method == "tabu":

        best_lcn, trajectory = tabu_search(
            samples_df,
            aggregate_table,
            nodes,
            mutation_type
        )

    else:
        raise ValueError("Unknown search method")

    # Edges are extracted only for evaluation
    best_edges = best_lcn["edges"] 

    bic_metrics = evaluate_interval_bic(trajectory)

    # SHD
    shd = compute_shd(lcn, best_edges)

    edge_metrics = compute_edge_metrics(lcn, best_edges, nodes)

    # KL Divergence 
    learned_bn = DiscreteBayesianNetwork(best_edges)
    learned_bn.add_nodes_from(nodes)
    learned_bn.fit(samples_df, estimator=MaximumLikelihoodEstimator)

    kl = compute_kl(lcn, learned_bn, samples_df)

    return {
        "config": {
            "size": size,
            "interval_width": interval_width,
            "width_dist_type": width_dist_type,
            "in_degree": in_degree,
            "num_samples": num_samples,
            "search_method": search_method,
            "mutation_type": mutation_type
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