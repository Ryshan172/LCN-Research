import copy
import itertools
import json
import math
import os
import random
import time
import zlib

import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork

from mutations.mutation_functions import extract_constraint_dependencies
from structure_learning.logic_aware_ghc import (
    logic_aware_greedy_hill_climbing,
    random_initial_structure,
)
from utils.data_saving import save_experiment_to_json
from workflows.rq2_experiments import (
    compute_edge_metrics,
    compute_kl,
    compute_shd,
    evaluate_interval_bic,
    generate_lcn,
    hill_climb,
    initialise_structure,
    sample_lcn,
    score_structure,
)


"""
RQ3: Can a logic-aware variable ordering optimisation procedure improve
LCN structure learning compared to established procedures?

Search procedures compared (scoring function and mutation operators are
identical across all of them):

    rq2_ghc:
        The RQ2 hill climber: one randomly sampled mutation per step,
        strict uphill acceptance, no tabu / restarts. It always starts
        from the empty graph and scores with "mid", so it is only run
        in configurations with initialisation="empty" and scoring="mid".

    ghc_tabu_rr:
        Established procedure: first-improvement GHC over a uniformly
        shuffled neighbourhood, with tabu list, plateau moves and
        random restarts.

    la_ghc:
        Logic-Aware Variable-Ordered GHC. Identical to ghc_tabu_rr except
        the neighbourhood is ordered by LANO. Any difference between
        ghc_tabu_rr and la_ghc is therefore attributable to the ordering.

Design:
    Paired. For each (size, in_degree, interval_width, repeat) a single
    ground-truth LCN is generated and sampled once. Every search method
    is then run on that same data, from the same initial structure, with
    the same (noisy) constraints and the same evaluation budget.

Problem difficulty is controlled by:
    - network size (up to 15 variables)
    - evaluation budget scaled to the neighbourhood size n(n-1)
    - random DAG initialisation (part of the RQ3 algorithm)
    - interval width and IBIC collapsing mode (low / mid / high)
    - noisy logical constraints supplied to the learner
"""


SEARCH_METHODS = ["rq2_ghc", "ghc_tabu_rr", "la_ghc"]


# ---------- DATA GENERATION -------------
def make_sampler_compatible(lcn):
    """
    Adds lexicographically ordered aliases of the credal set keys.

    WHY:
    The LCN generator keys credal sets by parents in edge order
    (X2 before X10), while sample_dataset() looks them up with
    sorted() parents (X10 before X2). For networks with 10+ variables
    a node with parents on both sides of X10 cannot be sampled.

    The alias points to the SAME credal set, so the distribution is
    unchanged. Existing generator and sampler code is not modified.
    """
    for node in lcn["nodes"]:
        # Parents in generator order vs. the sampler's sorted order
        parents = [u for u, v in lcn["edges"] if v == node]
        sorted_parents = sorted(parents)

        # Same order: nothing to fix for this node
        if parents == sorted_parents:
            continue

        # For every True/False combination of the parents ...
        for combo in itertools.product([True, False], repeat=len(parents)):
            assignment = dict(zip(parents, combo))

            # ... build the key the generator wrote and the key the
            # sampler will look up, and point both to the same credal set
            key = "[" + ", ".join(f"{p}={assignment[p]}" for p in parents) + "]"
            alias = "[" + ", ".join(f"{p}={assignment[p]}" for p in sorted_parents) + "]"

            lcn["credal_sets"][node][alias] = lcn["credal_sets"][node][key]

    return lcn


def generate_rq3_data(
    size,
    interval_width,
    width_dist_type,
    in_degree,
    num_samples,
    seed,
    max_attempts=5,
):
    """
    Generate and sample a ground-truth LCN with a reproducible seed.

    If generation fails validation, the next seed is tried. The seed
    actually used is returned so the run can be reproduced.
    """
    for attempt in range(max_attempts):
        # A different (but still reproducible) seed on each attempt
        data_seed = seed + attempt * 7919
        random.seed(data_seed)
        np.random.seed(data_seed)

        try:
            # Ground-truth LCN, made safe for the sampler, then sampled
            lcn = generate_lcn(size, interval_width, width_dist_type, in_degree)
            make_sampler_compatible(lcn)
            samples_df, aggregate_table = sample_lcn(lcn, num_samples)

            return lcn, samples_df, aggregate_table, data_seed

        except (RuntimeError, KeyError) as e:
            print(f"LCN generation attempt {attempt + 1} failed: {e}")

    raise RuntimeError(
        f"Could not generate LCN (size={size}, in_degree={in_degree}) "
        f"after {max_attempts} attempts."
    )


def corrupt_constraints(logical_constraints, true_edges, nodes, noise_rate, rng):
    """
    Replace a fraction of the logical constraints with incorrect ones.

    An incorrect constraint links two variables that are NOT adjacent
    in the ground-truth graph, with random truth values. This simulates
    imperfect domain knowledge. Only the constraints given to the
    learner are corrupted; the ground-truth LCN and its samples are not.

    Returns (noisy_constraints, indices_of_corrupted_constraints).
    """
    # Work on a copy so the true constraints are not changed
    noisy = copy.deepcopy(logical_constraints)

    # Round half up, so e.g. 0.25 of 2 constraints corrupts 1
    num_corrupt = min(
        len(noisy),
        math.floor(noise_rate * len(noisy) + 0.5)
    )

    if num_corrupt == 0:
        return noisy, []

    # Pairs that ARE linked in the true graph (either direction)
    adjacent = set()
    for a, b in true_edges:
        adjacent.add((a, b))
        adjacent.add((b, a))

    # Pairs that are NOT linked: a constraint here is wrong
    non_adjacent = [
        (a, b)
        for a in nodes
        for b in nodes
        if a != b and (a, b) not in adjacent
    ]

    # Choose which constraints to corrupt
    corrupted = sorted(rng.sample(range(len(noisy)), num_corrupt))

    # Replace each one with a rule between two unlinked variables
    for i in corrupted:
        a, b = rng.choice(non_adjacent)
        noisy[i] = {
            "if": {a: rng.choice([True, False])},
            "then": {b: rng.choice([True, False])},
        }

    return noisy, corrupted


def evaluation_budget(size, budget_factor=None, max_evaluations=None):
    """
    Number of score evaluations allowed per search.

    budget_factor scales the budget to the neighbourhood size n(n-1),
    so larger networks are not given proportionally more slack.
    """
    # n(n-1) is roughly the number of possible edits from one graph
    if budget_factor is not None:
        return max(10, int(budget_factor * size * (size - 1)))

    if max_evaluations is not None:
        return max_evaluations

    raise ValueError("Provide budget_factor or max_evaluations.")


def seed_for(*parts):
    """
    Deterministic seed from the experiment factors.
    """
    # e.g. seed_for(9, 2, 0.3, 0) always gives the same number
    return zlib.crc32("-".join(map(str, parts)).encode()) % (2 ** 31)


# ---------- SEARCH DISPATCH -------------
def method_applicable(search_method, initialisation, scoring):
    """
    rq2_ghc always starts from the empty graph and scores with "mid".
    """
    if search_method == "rq2_ghc":
        return initialisation == "empty" and scoring == "mid"

    return True


def run_search(
    search_method,
    samples_df,
    aggregate_table,
    initial_state,
    mutation_strategy,
    max_evaluations,
    seed,
    scoring="mid",
    rq2_mutation_type="edge_add",
    search_options=None,
):
    """
    Runs one search procedure and returns
    (best_lcn, trajectory, diagnostics).

    search_options:
        Extra keyword arguments for ghc_tabu_rr / la_ghc, e.g.
        {"allow_plateau": False, "max_restarts": 3}.
        Applied identically to both so they differ only in ordering.
    """

    if search_method == "rq2_ghc":
        # RQ2 hill_climb uses the global random module
        random.seed(seed)

        # One score evaluation per iteration, so max_iter = budget - 1
        # (the first evaluation scores the empty starting graph)
        best_lcn, trajectory = hill_climb(
            samples_df,
            aggregate_table,
            initial_state["nodes"],
            logical_constraints=initial_state["logical_constraints"],
            mutation_strategy=mutation_strategy,
            mutation_type=rq2_mutation_type,
            max_iter=max_evaluations - 1
        )

        # Evaluation at which the best score was first reached
        best_score = max(trajectory)

        diagnostics = {
            "evaluations": len(trajectory),
            "evaluations_to_best": trajectory.index(best_score) + 1,
            "best_score": best_score,
            "mutation_type": rq2_mutation_type,
        }

        return best_lcn, trajectory, diagnostics

    if search_method in ("ghc_tabu_rr", "la_ghc"):

        # Same algorithm for both; only the edit ordering differs
        return logic_aware_greedy_hill_climbing(
            samples_df,
            aggregate_table,
            initial_state,
            score_function=score_structure,
            mutation_strategy=mutation_strategy,
            ordering="lano" if search_method == "la_ghc" else "random",
            max_evaluations=max_evaluations,
            scoring=scoring,
            seed=seed,
            **(search_options or {}),
        )

    raise ValueError(f"Unknown search method: {search_method}")


# ------------ EVALUATION METRICS ------------------
def compute_constraint_satisfaction(logical_constraints, learned_edges):
    """
    Fraction of constraint dependencies (X -> Y) present as directed
    edges in the learned graph. Same structural interpretation as
    the constraint-aware mutations.
    """
    # Pairs X -> Y that the constraints say should be edges
    required = extract_constraint_dependencies(logical_constraints)

    if not required:
        return None

    learned = set(map(tuple, learned_edges))

    # Share of those pairs present in the learned graph
    return sum(pair in learned for pair in required) / len(required)


# ---------- EXPERIMENT RUNNER -------------
def run_rq3_experiment_steps(
    lcn,
    samples_df,
    aggregate_table,
    initial_state,
    search_method,
    mutation_strategy,
    max_evaluations,
    seed,
    scoring="mid",
    search_options=None,
):
    # ----------------------------------------------------
    # STEP 1: Information available to learner
    #
    # initial_state carries the nodes, the starting edges and the
    # (possibly noisy) logical constraints, but NOT the true graph.
    # ----------------------------------------------------
    nodes = lcn["nodes"]
    given_constraints = initial_state["logical_constraints"]

    # ----------------------------------------------------
    # STEP 2: Run structure learning (search phase)
    # ----------------------------------------------------
    start = time.perf_counter()

    best_lcn, trajectory, diagnostics = run_search(
        search_method,
        samples_df,
        aggregate_table,
        initial_state,
        mutation_strategy,
        max_evaluations,
        seed,
        scoring=scoring,
        search_options=search_options,
    )

    runtime_sec = time.perf_counter() - start

    best_edges = [tuple(edge) for edge in best_lcn["edges"]]

    # ----------------------------------------------------
    # STEP 3: Search behaviour metrics
    # ----------------------------------------------------
    bic_metrics = evaluate_interval_bic(trajectory)

    # ----------------------------------------------------
    # STEP 4: Structural comparison to ground truth
    # ----------------------------------------------------
    shd = compute_shd(lcn, best_edges)
    edge_metrics = compute_edge_metrics(lcn, best_edges, nodes)

    # Against the constraints the learner was given ...
    constraint_satisfaction = compute_constraint_satisfaction(
        given_constraints,
        best_edges
    )

    # ... and against the true constraints (differs under noise)
    true_constraint_satisfaction = compute_constraint_satisfaction(
        lcn["logical_constraints"],
        best_edges
    )

    # ----------------------------------------------------
    # STEP 5: Distributional evaluation (same setup as RQ2)
    #
    # KL is computed between:
    #   P = BN on the true structure fitted to LCN samples
    #   Q = BN on the learned structure fitted to LCN samples
    #
    # fit() without an estimator defaults to Maximum Likelihood
    # in both older and newer (>=1.1) pgmpy versions.
    # ----------------------------------------------------
    try:
        learned_bn = DiscreteBayesianNetwork(best_edges)
        learned_bn.add_nodes_from(nodes)
        learned_bn.fit(samples_df)

        true_bn = DiscreteBayesianNetwork(lcn["edges"])
        true_bn.add_nodes_from(nodes)
        true_bn.fit(samples_df)

        kl = compute_kl(true_bn, learned_bn, samples_df)

    except Exception as e:
        # e.g. RQ2 hill climbing is not guaranteed to return a DAG
        print(f"KL divergence not computed: {e}")
        kl = None

    # ----------------------------------------------------
    # STEP 6: Return full experiment results
    # ----------------------------------------------------
    return {
        "structure": {
            "initial_edges": [tuple(edge) for edge in initial_state["edges"]],
            "learned_edges": best_edges,
            "shd": shd
        },
        "edge_metrics": edge_metrics,
        "kl_divergence": kl,
        "interval_bic": bic_metrics,
        "constraint_satisfaction": constraint_satisfaction,
        "true_constraint_satisfaction": true_constraint_satisfaction,
        "runtime_sec": runtime_sec,
        "search_diagnostics": diagnostics,
        "trajectory": trajectory
    }


def rq3_experiment_run_variants(
    experiment_name="rq3_main",
    sizes=(5, 9, 12, 15),
    in_degrees=(1, 2, 3),
    interval_widths=(0.3,),
    scorings=("mid",),
    initialisations=("empty", "random"),
    constraint_noises=(0.0,),
    mutation_strategies=("standard", "constraint", "repair"),
    search_methods=SEARCH_METHODS,
    runs_per_config=10,
    budget_factor=2,
    max_evaluations=None,
    num_samples=200,
    width_dist_type="uniform",
    results_dir=None,
    base_seed=0,
    search_options=None,
):
    """
    Factorial RQ3 experiment.

    Data factors (one ground-truth LCN per combination and repeat):
        size, in_degree, interval_width

    Learner factors (crossed within each LCN, i.e. paired):
        initialisation, constraint_noise, scoring,
        mutation_strategy, search_method

    Every search method on the same LCN gets the same initial
    structure, the same constraints, the same evaluation budget
    and the same search seed.

    Note: in_degree above 3 is not used, since the existing LCN
    generator fails validation for high in-degrees.
    """
    results_dir = results_dir or f"results_{experiment_name}"
    run_counter = 1

    # Loop over the data factors: one LCN per combination and repeat
    for size, in_degree, interval_width in itertools.product(
        sizes,
        in_degrees,
        interval_widths
    ):

        # Evaluation budget for this network size
        budget = evaluation_budget(size, budget_factor, max_evaluations)

        for repeat_idx in range(runs_per_config):

            lcn, samples_df, aggregate_table, data_seed = generate_rq3_data(
                size,
                interval_width,
                width_dist_type,
                in_degree,
                num_samples,
                seed=base_seed + seed_for(size, in_degree, interval_width, repeat_idx),
            )

            nodes = lcn["nodes"]

            # Loop over the learner factors, all on this same LCN
            for initialisation in initialisations:

                # Same starting edges for every method and noise level
                empty_state = initialise_structure(nodes, logical_constraints=[])

                if initialisation == "random":
                    start_edges = random_initial_structure(
                        empty_state,
                        random.Random(seed_for(data_seed, "init"))
                    )["edges"]
                elif initialisation == "empty":
                    start_edges = []
                else:
                    raise ValueError(f"Unknown initialisation: {initialisation}")

                for constraint_noise in constraint_noises:

                    # Constraints the learner receives (possibly noisy)
                    given_constraints, corrupted = corrupt_constraints(
                        lcn["logical_constraints"],
                        lcn["edges"],
                        nodes,
                        constraint_noise,
                        random.Random(seed_for(data_seed, "noise", constraint_noise)),
                    )

                    # Learner's starting state: nodes, start edges and
                    # the given constraints (but not the true graph)
                    initial_state = initialise_structure(
                        nodes,
                        logical_constraints=given_constraints
                    )
                    initial_state["edges"] = list(start_edges)

                    # Same search seed for every method on this LCN
                    search_seed = seed_for(data_seed, "search")

                    for scoring, mutation_strategy, search_method in itertools.product(
                        scorings,
                        mutation_strategies,
                        search_methods
                    ):

                        # Skip rq2_ghc where it can't be configured
                        if not method_applicable(search_method, initialisation, scoring):
                            continue

                        print(
                            f"{experiment_name} run {run_counter} | "
                            f"size={size}, in_degree={in_degree}, "
                            f"width={interval_width}, init={initialisation}, "
                            f"noise={constraint_noise}, scoring={scoring}, "
                            f"strategy={mutation_strategy}, "
                            f"search={search_method}, repeat={repeat_idx + 1}"
                        )

                        results = run_rq3_experiment_steps(
                            lcn,
                            samples_df,
                            aggregate_table,
                            initial_state,
                            search_method=search_method,
                            mutation_strategy=mutation_strategy,
                            max_evaluations=budget,
                            seed=search_seed,
                            scoring=scoring,
                            search_options=search_options,
                        )

                        experiment_obj = {
                            "experiment": experiment_name,
                            "run_id": run_counter,
                            "repeat": repeat_idx + 1,
                            "params": {
                                "size": size,
                                "interval_width": interval_width,
                                "width_dist_type": width_dist_type,
                                "in_degree": in_degree,
                                "num_samples": num_samples,
                                "initialisation": initialisation,
                                "constraint_noise": constraint_noise,
                                "constraints_corrupted": len(corrupted),
                                "scoring": scoring,
                                "mutation_strategy": mutation_strategy,
                                "search_method": search_method,
                                "max_evaluations": budget,
                                "budget_factor": budget_factor,
                                "data_seed": data_seed,
                                "search_seed": search_seed,
                                "search_options": search_options or {},
                            },
                            "ground_truth": {
                                "edges": lcn["edges"],
                                "logical_constraints": lcn["logical_constraints"],
                            },
                            "given_constraints": given_constraints,
                            "corrupted_constraint_indices": corrupted,
                            "results": results
                        }

                        # One JSON file per run
                        save_experiment_to_json(
                            experiment_obj,
                            f"{experiment_name}_run_{run_counter}",
                            results_dir
                        )

                        run_counter += 1


# ---------- EXPERIMENT PRESETS -------------
def rq3_main_experiments():
    """
    Main comparison: network size x in-degree x initialisation,
    with a budget scaled to the neighbourhood size.
    """
    rq3_experiment_run_variants(
        experiment_name="rq3_main",
        sizes=(5, 9, 12, 15),
        in_degrees=(1, 2, 3),
        initialisations=("empty", "random"),
    )


def rq3_sensitivity_experiments():
    """
    Sensitivity to interval width and IBIC collapsing mode.
    """
    rq3_experiment_run_variants(
        experiment_name="rq3_sensitivity",
        sizes=(9, 12),
        in_degrees=(2,),
        interval_widths=(0.3, 0.5),
        scorings=("low", "mid", "high"),
        initialisations=("random",),
    )


def rq3_robustness_experiments():
    """
    Robustness to incorrect logical constraints. noise=1.0 means
    every constraint given to the learner is wrong.
    """
    rq3_experiment_run_variants(
        experiment_name="rq3_robustness",
        sizes=(9, 12),
        in_degrees=(2,),
        initialisations=("random",),
        constraint_noises=(0.0, 0.5, 1.0),
    )


def summarise_rq3_results(
    results_dir="results_rq3_main",
    output_csv=None
):
    """
    Reads RQ3 experiment JSON files and summarises them into a CSV,
    one row per run.
    """
    output_csv = output_csv or f"{os.path.basename(os.path.normpath(results_dir))}_summary.csv"

    rows = []

    # Read every run's JSON file into one row
    for file in os.listdir(results_dir):
        if not file.endswith(".json"):
            continue

        with open(os.path.join(results_dir, file), "r") as f:
            data = json.load(f)

        params = data.get("params", {})
        results = data.get("results", {})
        structure = results.get("structure", {})
        edge_metrics = results.get("edge_metrics", {})
        shd = structure.get("shd", {})
        ibic = results.get("interval_bic", {})
        diagnostics = results.get("search_diagnostics", {})
        trajectory = results.get("trajectory", [])

        rows.append({
            # identifiers
            "experiment": data.get("experiment"),
            "run_id": data.get("run_id"),
            "repeat": data.get("repeat"),
            "data_seed": params.get("data_seed"),

            # experiment params
            "size": params.get("size"),
            "in_degree": params.get("in_degree"),
            "interval_width": params.get("interval_width"),
            "num_samples": params.get("num_samples"),
            "initialisation": params.get("initialisation"),
            "constraint_noise": params.get("constraint_noise"),
            "constraints_corrupted": params.get("constraints_corrupted"),
            "scoring": params.get("scoring"),
            "mutation_strategy": params.get("mutation_strategy"),
            "search_method": params.get("search_method"),
            "max_evaluations": params.get("max_evaluations"),

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
            "initial_edges": len(structure.get("initial_edges", [])),

            # probabilistic / scoring metrics
            "kl_divergence": results.get("kl_divergence"),
            "ibic_best_score": ibic.get("best_score"),
            "ibic_stability": ibic.get("stability"),
            "constraint_satisfaction": results.get("constraint_satisfaction"),
            "true_constraint_satisfaction": results.get("true_constraint_satisfaction"),

            # search efficiency
            "evaluations": diagnostics.get("evaluations"),
            "evaluations_to_best": diagnostics.get("evaluations_to_best"),
            "restarts": diagnostics.get("restarts"),
            "accepted_uphill": diagnostics.get("accepted_uphill"),
            "accepted_neutral": diagnostics.get("accepted_neutral"),
            "termination_reason": diagnostics.get("termination_reason"),
            "runtime_sec": results.get("runtime_sec"),
            "trajectory_len": len(trajectory),
        })

    df = pd.DataFrame(rows)

    df = df.sort_values(by=["run_id"]).reset_index(drop=True)

    df.to_csv(output_csv, index=False)

    # Quick overview: average results per method and strategy
    print(f"Saved summary to: {output_csv}")
    print(
        df.groupby(["search_method", "mutation_strategy"])[
            ["ibic_best_score", "shd", "f1", "evaluations_to_best"]
        ].mean()
    )

    return df
