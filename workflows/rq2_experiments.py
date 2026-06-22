import pandas as pd
import itertools
import numpy as np
import copy

from lcn_functions.model import create_lcn
from mutations.mutation_functions import standard_mutation, contraint_aware_mutation, post_mutation_contraint_repair
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

#----- Helpers ------------------------------------------
def build_constraint_index(logical_constraints):
    """
    Precompute constraint → variable index.

    WHY:
    This enforces Assumption 2 (locality) in practice.

    Instead of scanning ALL constraints per mutation,
    we do O(1)-style lookup per affected node.
    """

    index = {}

    for rule in logical_constraints:

        variables = set(rule["if"].keys()) | set(rule["then"].keys())

        for v in variables:
            if v not in index:
                index[v] = []
            index[v].append(rule)

    return index


# ----------- LCN GENERATION ------------------------------------
def generate_lcn(size, interval_width, width_dist_type, in_degree):
    return create_lcn(size, interval_width, width_dist_type, in_degree)


def sample_lcn(lcn, num_samples):
    samples = sample_dataset(lcn, num_samples)
    samples_df = pd.DataFrame(samples)

    aggregate_table = credal_aggregate_intervals(samples, lcn)

    return samples_df, aggregate_table


# -------- INITIAL STRUCTURE -------------------------
def initialise_structure(
    nodes,
    logical_constraints=None,
    strategy="empty"
):
    """
    Full LCN state initialisation.

    IMPORTANT CHANGE:
    We now build a constraint index so locality is REAL,
    not just assumed in theory.

    NEW:
    Logical constraints can now be supplied from the
    generated LCN.

    This allows constraint-aware and repair mutations
    to access domain knowledge while still starting
    from an empty graph structure.
    """

    if logical_constraints is None:
        logical_constraints = []

    return {
        "nodes": list(nodes),

        # learner still starts from empty structure
        "edges": [],

        "credal_sets": {},

        # domain knowledge supplied to learner
        "logical_constraints": logical_constraints,

        # locality support structure
        "constraint_index": build_constraint_index(
            logical_constraints
        )
    }


# ----- MUTATION OPERATORS ------------
def mutate_lcn(lcn_state, strategy="standard", mutation_type="edge_add"):
    
    constraint_index = lcn_state.get("constraint_index", None)

    if strategy == "standard":
        return standard_mutation(lcn_state, mutation_type)

    elif strategy == "constraint":
        return contraint_aware_mutation(
            lcn_state,
            constraint_index,
            mutation_type
        )

    elif strategy == "repair":
        return post_mutation_contraint_repair(
            lcn_state,
            constraint_index,
            mutation_type
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    

# -------- SCORE FUNCTION ------------------
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


# -------- HILL CLIMBING CORE -------------------
def hill_climb(
    samples_df,
    aggregate_table,
    nodes,
    logical_constraints=None,
    mutation_strategy="standard",
    mutation_type="edge_add",
    max_iter=100
):
    """
    Standard greedy hill climbing over FULL LCN state.
    """

    # Updated to pass logical constraints as well
    current = initialise_structure(
        nodes,
        logical_constraints=logical_constraints
    )

    current_score = score_structure(current, samples_df, aggregate_table)

    best = copy.deepcopy(current)
    best_score = current_score

    trajectory = [current_score]

    for _ in range(max_iter):

        # correct mutation interface
        candidate = mutate_lcn(
            current,
            strategy=mutation_strategy,
            mutation_type=mutation_type
        )

        candidate_score = score_structure(candidate, samples_df, aggregate_table)

        if candidate_score > current_score:
            current = candidate
            current_score = candidate_score

            if candidate_score > best_score:
                best = copy.deepcopy(candidate)
                best_score = candidate_score

        trajectory.append(current_score)

    return best, trajectory


# ---------- RANDOM RESTART ------------------------
def random_restart_hill_climb(
    samples_df,
    aggregate_table,
    nodes,
    logical_constraints=None,
    mutation_type="edge_add",
    mutation_strategy="standard",
    n_restarts=10,
    max_iter=50
):
    """
    Random restart hill climbing over full LCN state.

    FIX:
    Now correctly forwards BOTH:
    - mutation_strategy (standard / constraint / repair)
    - mutation_type (edge_add / delete / flip)
    """

    # NOTE: Needs to be updated like Hill Climbing (TODO Later)

    best_overall = None
    best_score = -np.inf
    full_trajectory = []

    for _ in range(n_restarts):

        # Updated for updated Hill Climb
        result, traj = hill_climb(
            samples_df,
            aggregate_table,
            nodes,
            logical_constraints=logical_constraints,
            mutation_strategy=mutation_strategy,
            mutation_type=mutation_type,
            max_iter=max_iter
        )

        final_score = traj[-1]
        full_trajectory.extend(traj)

        if final_score > best_score:
            best_score = final_score
            best_overall = result

    return best_overall, full_trajectory


# ------------- TABU SEARCH ------------------
def tabu_search(
    samples_df,
    aggregate_table,
    nodes,
    logical_constraints=None,
    mutation_strategy="standard",
    mutation_type="edge_add",
    tabu_size=10,
    max_iter=100
):
    """
    Tabu search over FULL LCN state.
    """

    # Updated because of logical constraints 
    current = initialise_structure(
    nodes,
        logical_constraints=logical_constraints
    )

    current_score = score_structure(current, samples_df, aggregate_table)

    best = copy.deepcopy(current)
    best_score = current_score

    tabu_list = []
    trajectory = [current_score]

    for _ in range(max_iter):

        candidate = mutate_lcn(
            current,
            strategy=mutation_strategy,
            mutation_type=mutation_type
        )

        # Tabu check (structure-level comparison)
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


# ------------ EVALUATION METRICS ------------------
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


# ---------- EXPERIMENT RUNNER -------------
def run_experiment_steps(
    size,
    interval_width,
    width_dist_type,
    in_degree,
    num_samples,
    search_method="hill_climbing",
    mutation_type="edge_add",
    mutation_strategy="standard"
):
    # ----------------------------------------------------
    # STEP 1: Generate a ground-truth LCN (data-generating model)
    # ----------------------------------------------------
    lcn = generate_lcn(size, interval_width, width_dist_type, in_degree)

    # ----------------------------------------------------
    # STEP 2: Sample data from the LCN
    # - produces observed dataset (samples_df)
    # - produces interval/credal aggregation statistics
    # ----------------------------------------------------
    samples_df, aggregate_table = sample_lcn(lcn, num_samples)

    # ----------------------------------------------------
    # STEP 3: Extract information available to learner
    #
    # nodes:
    #     variables over which structure learning occurs
    #
    # logical_constraints:
    #     domain knowledge supplied to constraint-aware
    #     and repair mutation strategies
    #
    # IMPORTANT:
    # The learner receives logical constraints but NOT
    # the true graph structure.
    # ----------------------------------------------------
    nodes = list(lcn.nodes())

    logical_constraints = getattr(
        lcn,
        "logical_constraints",
        []
    )

    # ----------------------------------------------------
    # STEP 4: Run structure learning (search phase)
    # - chooses optimisation strategy (HC / RR / Tabu)
    # - returns best learned structure + score trajectory
    # ----------------------------------------------------
    if search_method == "hill_climbing":
        
        # Hill Climbing call Including logical constraints as well
        best_lcn, trajectory = hill_climb(
            samples_df,
            aggregate_table,
            nodes,
            logical_constraints=logical_constraints,
            mutation_strategy=mutation_strategy,
            mutation_type=mutation_type
        )

    elif search_method == "random_restart":
        # Random Restart call including Logical Constraints 
        best_lcn, trajectory = random_restart_hill_climb(
            samples_df,
            aggregate_table,
            nodes,
            logical_constraints=logical_constraints,
            mutation_type=mutation_type,
            mutation_strategy=mutation_strategy
        )

    elif search_method == "tabu":
        # Tabu search call with logical constraints added 
        best_lcn, trajectory = tabu_search(
            samples_df,
            aggregate_table,
            nodes,
            logical_constraints=logical_constraints,
            mutation_strategy=mutation_strategy,
            mutation_type=mutation_type
        )

    else:
        raise ValueError("Unknown search method")

    # ----------------------------------------------------
    # STEP 5: Extract learned graph structure (edges only)
    # ----------------------------------------------------
    best_edges = best_lcn["edges"]

    # ----------------------------------------------------
    # STEP 6: Evaluate search behaviour (optimization metrics)
    # - how good score was
    # - how fast it converged
    # - stability of search trajectory
    # ----------------------------------------------------
    bic_metrics = evaluate_interval_bic(trajectory)

    # ----------------------------------------------------
    # STEP 7: Structural comparison to ground truth
    # - SHD measures edge-level structural error
    # - edge_metrics gives precision/recall/F1 view
    # ----------------------------------------------------
    shd = compute_shd(lcn, best_edges)
    edge_metrics = compute_edge_metrics(lcn, best_edges, nodes)

    # ----------------------------------------------------
    # STEP 8: Fit a Bayesian Network on learned structure
    # - converts learned edges into probabilistic model
    # - fits CPTs using Maximum Likelihood Estimation
    # ----------------------------------------------------
    learned_bn = DiscreteBayesianNetwork(best_edges)
    learned_bn.add_nodes_from(nodes)
    learned_bn.fit(samples_df, estimator=MaximumLikelihoodEstimator)

    # ----------------------------------------------------
    # STEP 9: Distributional evaluation
    # - KL divergence compares learned vs true distribution
    # ----------------------------------------------------
    kl = compute_kl(lcn, learned_bn, samples_df)

    # ----------------------------------------------------
    # STEP 10: Return full experiment results
    # ----------------------------------------------------
    return {
        "config": {
            "size": size,
            "interval_width": interval_width,
            "width_dist_type": width_dist_type,
            "in_degree": in_degree,
            "num_samples": num_samples,
            "search_method": search_method,
            "mutation_type": mutation_type,
            "mutation_strategy": mutation_strategy
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