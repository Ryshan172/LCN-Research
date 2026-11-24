import pandas as pd
import itertools
import random
import numpy as np
import copy

import os
import json
import pandas as pd

from lcn_functions.model import create_lcn
from mutations.mutation_functions import (
    constraint_aware_mutation, extract_constraint_dependencies,
    post_mutation_contraint_repair, standard_mutation, valid_dag,
)
from sampler_functions.contingency_sampler import sample_dataset, credal_aggregate_intervals
from metric_functions.structural_hamming_distance import structural_hamming_distance_compare
from metric_functions.kl_divergence import kl_divergence_from_samples
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

from scoring_functions.interval_bic_derivation import compute_interval_BIC
from scoring_functions.scoring_helpers import regroup_for_parents
from utils.data_saving import save_experiment_to_json


"""
RQ2: Heuristic Structure Learning over LCNs using Interval BIC
- mutation strategies
- hill climbing / random restart / tabu search
- constraint-aware structure learning
"""

#----- Helpers ------------------------------------------
def build_constraint_index(logical_constraints):
    """Precompute the variable-to-rule index used by constraint-aware mutations.

    Register each rule under every variable in its "if" and "then" parts.
    After a single-edge edit, the two endpoints identify which lists to
    retrieve. This implements local rule lookup without scanning the full
    constraint set each time. Reading a retrieved list still takes time
    proportional to the number of rules it contains.
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
def initialise_structure(nodes, logical_constraints=None, strategy="empty", max_in_degree=None,
                         initial_edges=None):
    """Build the initial graph state and rule index for structure learning.

    The state contains the supplied nodes, starting edges, an empty
    credal_sets dictionary, the domain rules and their variable-to-rule
    index. Both constrained strategies receive the supplied rules directly;
    the learner does not need the generating graph to extract required arcs.

    With no explicit starting graph, the baseline and the "empty" strategy start with no
    edges. The "constraint" and "repair" strategies start with exactly the
    rule-derived arcs so the first incumbent already satisfies the structural
    prior. Raise ValueError if the starting edges cannot form a DAG on the
    given nodes within max_in_degree. RQ2 carries this limit into mutation;
    the default empty-state call without a cap keeps the legacy state format.
    """
    nodes = list(nodes)
    rules = logical_constraints or []
    # Changed: Validate the supplied prior and any optional starting arcs before scoring.
    constrained = strategy in ("constraint", "repair")
    required = extract_constraint_dependencies(rules) if constrained else set()
    # Changed: An experiment may supply a feasible starting DAG with optional edges.
    edges = sorted(required) if initial_edges is None else [tuple(edge) for edge in initial_edges]
    if not required.issubset(edges):
        raise ValueError("The starting graph is missing required arcs")
    if not valid_dag(nodes, edges, max_in_degree):
        raise ValueError(
            f"No feasible starting DAG on the supplied nodes with parent cap "
            f"{max_in_degree} and initial edges {edges}"
        )

    state = {
        "nodes": nodes,
        "edges": edges,
        "credal_sets": {},
        "logical_constraints": rules,
        "constraint_index": build_constraint_index(rules),
    }
    # Changed: Carry the RQ2 parent cap into mutations while preserving the legacy empty-state format.
    if strategy != "empty" or max_in_degree is not None:
        state["max_in_degree"] = max_in_degree
    return state


def experiment_start(nodes, rules, strategy, mutation_type, max_in_degree, seed):
    """Choose a start from the nodes and prior alone; return its edges and move availability."""
    base = initialise_structure(nodes, rules, strategy, max_in_degree)["edges"]
    pairs = list(itertools.permutations(nodes, 2))
    if mutation_type == "edge_add":
        available = any(valid_dag(nodes, base + [pair], max_in_degree) for pair in pairs)
        return base, available
    if mutation_type not in ("edge_delete", "edge_flip", "edge_reverse"):
        raise ValueError(f"Unknown mutation type: {mutation_type}")

    # Changed: Add up to n-1 optional arcs so fixed deletion/reversal runs can move.
    # For a reversal both orientations must be legal at insertion: the last arc
    # inserted can therefore still be reversed in the completed starting graph.
    random.Random(seed).shuffle(pairs)
    edges = list(base)
    for u, v in pairs:
        if not valid_dag(nodes, edges + [(u, v)], max_in_degree):
            continue
        if mutation_type != "edge_delete" and not valid_dag(nodes, edges + [(v, u)], max_in_degree):
            continue
        edges.append((u, v))
        if len(edges) - len(base) == len(nodes) - 1:
            break
    return edges, len(edges) > len(base)


def paired_trials(sizes, in_degrees, strategies, mutation_types, repeats, make_model, num_samples):
    """Share the generating model, samples and start seed within each strategy comparison."""
    # Changed: Retain the existing repeat schedule and run IDs, pairing inputs by construction.
    first_id = 1
    for size, in_degree in itertools.product(sizes, in_degrees):
        for repeat_idx in range(repeats):
            model = make_model(size, in_degree)
            samples, aggregate = sample_lcn(model, num_samples)
            trial_data = (model, samples, aggregate)
            seed = random.getrandbits(32)
            mutation_type = mutation_types[repeat_idx % len(mutation_types)]
            for index, strategy in enumerate(strategies):
                run_id = first_id + index * repeats + repeat_idx
                yield run_id, size, in_degree, repeat_idx, strategy, mutation_type, trial_data, seed
        first_id += len(strategies) * repeats


# ----- MUTATION OPERATORS ------------
def mutate_lcn(lcn_state, strategy="standard", mutation_type="edge_add"):
    
    constraint_index = lcn_state.get("constraint_index", None)

    if strategy == "standard":
        return standard_mutation(lcn_state, mutation_type)

    elif strategy == "constraint":
        return constraint_aware_mutation(
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
def score_structure(
    lcn_state,
    samples_df,
    aggregate_table,
    scoring="mid"
):
    """
    Structure-aware interval BIC score.

    Equivalent to the pgmpy IntervalBICScore,
    but for the custom LCN hill climber.
    """

    total_score = 0.0

    nodes = lcn_state["nodes"]
    edges = lcn_state["edges"]

    for node in nodes:

        parents = [
            parent
            for parent, child in edges
            if child == node
        ]

        df_regrouped = regroup_for_parents(
            aggregate_table,
            node,
            parents
        )

        interval_bic = compute_interval_BIC(
            df_regrouped
        )

        bic_lower, bic_upper = interval_bic[node]

        if scoring == "low":
            score = bic_lower

        elif scoring == "high":
            score = bic_upper

        elif scoring == "mid":
            score = (bic_lower + bic_upper) / 2

        else:
            raise ValueError(
                "scoring must be 'low', 'mid', or 'high'"
            )

        total_score += float(score)

    return total_score


# -------- HILL CLIMBING CORE -------------------
def hill_climb(
    samples_df,
    aggregate_table,
    nodes,
    logical_constraints=None,
    mutation_strategy="standard",
    mutation_type="edge_add",
    max_iter=100,
    scoring="mid",
    max_in_degree=None,
    record_noops=True,
    initial_edges=None,
):
    """
    Greedy hill climbing using the selected
    Interval BIC scoring mode: low, mid, or high. By default the trajectory has
    an entry per iteration. With record_noops=False it has an entry per score call.
    """

    # Changed: Initialize the search with its chosen mutation policy and parent cap.
    current = initialise_structure(nodes, logical_constraints, mutation_strategy, max_in_degree,
                                   initial_edges=initial_edges)

    current_score = score_structure(
        current,
        samples_df,
        aggregate_table,
        scoring=scoring
    )

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

        # Changed: Reuse the current score when mutation returns the incumbent.
        if candidate is current:
            if record_noops:
                trajectory.append(current_score)
            continue
        candidate_score = score_structure(
            candidate,
            samples_df,
            aggregate_table,
            scoring=scoring
        )

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
    max_iter=50,
    max_in_degree=None,
    initial_edges=None,
):
    """
    Random restart hill climbing over full LCN state.

    FIX:
    Now correctly forwards BOTH:
    - mutation_strategy (standard / constraint / repair)
    - mutation_type (edge_add / delete / flip)
    """

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
            max_iter=max_iter,
            # Changed: Enforce the same parent cap at every restart.
            max_in_degree=max_in_degree,
            initial_edges=initial_edges,
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
    max_iter=100,
    max_in_degree=None,
    initial_edges=None,
):
    """
    Tabu search over FULL LCN state.
    """

    # Changed: Initialize the search with its chosen mutation policy and parent cap.
    current = initialise_structure(nodes, logical_constraints, mutation_strategy, max_in_degree,
                                   initial_edges=initial_edges)

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

        # Changed: Reuse the current score when mutation returns the incumbent.
        if candidate is current:
            trajectory.append(current_score)
            continue

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
    """
    Converts your ground-truth LCN from a dictionary containing an "edges" list 
    into the edge-list format that the SHD function expects, instead of incorrectly 
    treating the dictionary as a graph object with an .edges() method.
    """
    true_edges = [
        tuple(edge)
        for edge in true_model["edges"]
    ]

    return structural_hamming_distance_compare(
        true_edges,
        learned_edges
    )


def compute_edge_metrics(true_model, learned_edges, nodes):
    """
    Ground-truth LCN is represented as:
    {"nodes": [...], "edges": [[parent, child], ...]}
    Convert edge lists to tuples for set-based metric calculations.
    true_edges = set(map(tuple, true_model["edges"]))
    """

    true_edges = set(map(tuple, true_model["edges"]))
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
    mutation_strategy="standard",
    trial_data=None,
    start_seed=None,
):
    # Changed: A grid repeat shares the same data across all mutation strategies.
    if trial_data is None:
        lcn = generate_lcn(size, interval_width, width_dist_type, in_degree)
        samples_df, aggregate_table = sample_lcn(lcn, num_samples)
    else:
        lcn, samples_df, aggregate_table = trial_data

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
    # The learner receives logical constraints but NOT
    # the true graph structure.
    # ----------------------------------------------------
    # nodes = list(lcn.nodes())
    nodes = lcn["nodes"]

    # Changed: Read rules from the generated model dictionary and pass the configured cap to each search.
    logical_constraints = lcn.get("logical_constraints", [])
    if start_seed is None:
        start_seed = random.getrandbits(32)
    initial_edges, move_available = experiment_start(
        nodes, logical_constraints, mutation_strategy, mutation_type, in_degree, start_seed
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
            max_in_degree=in_degree,
            initial_edges=initial_edges,
            mutation_strategy=mutation_strategy,
            mutation_type=mutation_type
        )

        print("DONE 4")

    elif search_method == "random_restart":
        # Random Restart call including Logical Constraints 
        best_lcn, trajectory = random_restart_hill_climb(
            samples_df,
            aggregate_table,
            nodes,
            logical_constraints=logical_constraints,
            max_in_degree=in_degree,
            initial_edges=initial_edges,
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
            max_in_degree=in_degree,
            initial_edges=initial_edges,
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
    print("DONE 6.5")
    edge_metrics = compute_edge_metrics(lcn, best_edges, nodes)
    print("DONE 7")

    # ----------------------------------------------------
    # STEP 8: Fit a Bayesian Network on learned structure
    # - converts learned edges into probabilistic model
    # - fits CPTs using Maximum Likelihood Estimation
    # ----------------------------------------------------
    learned_bn = DiscreteBayesianNetwork(best_edges)
    learned_bn.add_nodes_from(nodes)
    learned_bn.fit(samples_df, estimator=MaximumLikelihoodEstimator)

    print("DONE 8")

    # ----------------------------------------------------
    # STEP 9: Distributional evaluation (VALID KL SETUP)
    # ----------------------------------------------------
    # For KL computation, both models must be in pgmpy BN format
    # (i.e., support get_cpds()).
    #
    # Converting the LCN-induced samples into a pgmpy BayesianNetwork
    # using MLE parameter fitting, enabling a valid comparison.
    #
    # KL is computed between:
    #   P = BN fitted to LCN-generated samples
    #   Q = learned BN
    # ----------------------------------------------------

    true_bn = DiscreteBayesianNetwork(lcn["edges"])
    true_bn.add_nodes_from(nodes)

    # Fit CPDs from LCN-generated samples (empirical projection of LCN)
    true_bn.fit(samples_df, estimator=MaximumLikelihoodEstimator)

    kl = compute_kl(true_bn, learned_bn, samples_df)

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
            # Changed: Distinguish an immobile start from a feasible edit rejected by the score.
            "start_seed": start_seed,
            "initial_edges": initial_edges,
            "initial_move_available": move_available,
            "learned_edges": best_edges,
            "shd": shd
        },
        "edge_metrics": edge_metrics,
        "kl_divergence": kl,
        "interval_bic": bic_metrics,
        "trajectory": trajectory
    }



"""
This experiment uses a factorial design over:
    - network size
    - mutation strategy (standard, constraint, repair)

The mutation type (edge_add, edge_delete, edge_reverse)
is intentionally NOT included in the full grid to avoid
combinatorial explosion in the number of runs.

Instead, mutation types are STRATIFIED within each
(size, mutation_strategy) configuration using a
round-robin scheme:

    mutation_type = mutation_types[repeat_idx % len(...)]

This ensures:
    - balanced coverage of all mutation operators
    - no bias toward a single mutation type
    - controlled stochastic variation without increasing
      the total number of experiments

In this setup, mutation_type is treated as a controlled
nuisance factor, while size and mutation_strategy are
the primary experimental variables of interest.

Less variability with LCNs because mutations strategies are what 
are really being tested
"""
def rq2_experiment_run_variants_simple():

    # FIXED PARAMETERS
    interval_width = 0.3
    width_dist_type = "uniform"
    num_samples = 200

    search_method = "hill_climbing"

    # VARIABLES OF INTEREST
    sizes = [5, 7, 9]

    in_degrees = [1, 2, 3, 4, 5]

    mutation_strategies = [
        "standard",
        "constraint",
        "repair"
    ]

    mutation_types = [
        "edge_add",
        "edge_delete",
        "edge_reverse"
    ]

    runs_per_config = 10

    def make_model(size, in_degree):
        return generate_lcn(size, interval_width, width_dist_type, in_degree)

    trials = paired_trials(sizes, in_degrees, mutation_strategies, mutation_types,
                          runs_per_config, make_model, num_samples)
    for (run_counter, size, in_degree, repeat_idx, mutation_strategy,
         mutation_type, trial_data, start_seed) in trials:
        print(
            f"Run {run_counter} | "
            f"size={size}, "
            f"in_degree={in_degree}, "
            f"strategy={mutation_strategy}, "
            f"mutation={mutation_type}, "
            f"repeat={repeat_idx + 1}"
        )

        results = run_experiment_steps(
            size=size,
            interval_width=interval_width,
            width_dist_type=width_dist_type,
            in_degree=in_degree,
            num_samples=num_samples,
            search_method=search_method,
            mutation_type=mutation_type,
            mutation_strategy=mutation_strategy,
            trial_data=trial_data,
            start_seed=start_seed,
        )

        experiment_obj = {
            "run_id": run_counter,
            "repeat": repeat_idx + 1,
            "params": {
                "size": size,
                "interval_width": interval_width,
                "width_dist_type": width_dist_type,
                "in_degree": in_degree,
                "num_samples": num_samples,
                "mutation_type": mutation_type,
                "mutation_strategy": mutation_strategy,
                "search_method": search_method
            },
            "results": results
        }

        save_experiment_to_json(
            experiment_obj,
            f"rq2_run_{run_counter}",
            "results_rq2"
        )


"""
CHECK
- mutations cannot create cycles,
- delete/reverse experiments start with legal optional arcs whenever possible,
- regroup_for_parents() truly reproduces the same regrouping used in the original
"""


def summarise_rq2_results(
    results_dir="results_rq2",
    output_csv="rq2_results_summary.csv"
):
    """
    Reads interval learning experiment JSON files and summarises into a CSV.

    Output columns include:
    run_id, repeat, params,
    edge metrics (precision, recall, accuracy, f1),
    structure metrics (SHD, missing, extra, orientation errors),
    scoring metrics (KL divergence, IBIC stats),
    trajectory summary stats.
    """

    rows = []

    for file in os.listdir(results_dir):
        if not file.endswith(".json"):
            continue

        file_path = os.path.join(results_dir, file)

        with open(file_path, "r") as f:
            data = json.load(f)

        params = data.get("params", {})
        results = data.get("results", {})
        structure = results.get("structure", {})
        edge_metrics = results.get("edge_metrics", {})
        shd = structure.get("shd", {})
        ibic = results.get("interval_bic", {})
        trajectory = results.get("trajectory", [])

        # trajectory summaries
        traj_final = trajectory[-1] if trajectory else None
        traj_max = max(trajectory) if trajectory else None
        traj_len = len(trajectory)

        row = {
            # identifiers
            "run_id": data.get("run_id"),
            # Changed: Older result files have no starting-graph diagnostics.
            "initial_move_available": structure.get("initial_move_available"),
            "initial_edge_count": len(structure["initial_edges"]) if "initial_edges" in structure else None,
            "repeat": data.get("repeat"),

            # experiment params
            "size": params.get("size"),
            "interval_width": params.get("interval_width"),
            "width_dist_type": params.get("width_dist_type"),
            "in_degree": params.get("in_degree"),
            "num_samples": params.get("num_samples"),
            "mutation_type": params.get("mutation_type"),
            "mutation_strategy": params.get("mutation_strategy"),
            "search_method": params.get("search_method"),

            # edge performance
            "precision": edge_metrics.get("precision"),
            "recall": edge_metrics.get("recall"),
            "accuracy": edge_metrics.get("accuracy"),
            "f1": edge_metrics.get("f1"),

            # structure error metrics
            "shd": shd.get("shd"),
            "missing_edges": len(shd.get("missing", [])),
            "extra_edges": len(shd.get("extra", [])),
            "orientation_errors": shd.get("orientation_errors"),

            # probabilistic / scoring metrics
            "kl_divergence": results.get("kl_divergence"),
            "ibic_best_score": ibic.get("best_score"),
            "ibic_convergence_speed": ibic.get("convergence_speed"),
            "ibic_stability": ibic.get("stability"),

            # trajectory summaries
            "trajectory_len": traj_len,
            "trajectory_final": traj_final,
            "trajectory_max": traj_max,
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by=["size", "interval_width", "num_samples", "run_id", "repeat"]
    ).reset_index(drop=True)

    df.to_csv(output_csv, index=False)

    print(f"Saved summary to: {output_csv}")
    print(df.head())

    return df