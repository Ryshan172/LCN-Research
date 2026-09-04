import itertools

from utils.data_saving import save_experiment_to_json
from workflow_helpers.generate_lcn_from_csv import generate_and_save_lcn
from workflows.rq2_experiments import compute_edge_metrics, compute_kl, compute_shd, evaluate_interval_bic, generate_lcn, hill_climb, random_restart_hill_climb, sample_lcn, tabu_search
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

"""
Similar setup to RQ2 Experiments

Except that LCN is generated from medical data
"""


# ---------- EXPERIMENT RUNNER -------------

def run_rq2_med_experiment_steps(
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
    # Still to complete
    lcn = generate_and_save_lcn()

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
    # The learner receives logical constraints but NOT
    # the true graph structure.
    # ----------------------------------------------------
    # nodes = list(lcn.nodes())
    nodes = lcn["nodes"]

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

        print("DONE 4")

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
            "learned_edges": best_edges,
            "shd": shd
        },
        "edge_metrics": edge_metrics,
        "kl_divergence": kl,
        "interval_bic": bic_metrics,
        "trajectory": trajectory
    }


def rq2_med_experiments_variation_run():

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

    run_counter = 1
    all_experiments = []

    # GRID
    for size, in_degree, mutation_strategy in itertools.product(
        sizes,
        in_degrees,
        mutation_strategies
    ):

        # stratified cycle of mutation types per configuration
        for repeat_idx in range(runs_per_config):

            mutation_type = mutation_types[
                repeat_idx % len(mutation_types)
            ]

            print(
                f"Run {run_counter} | "
                f"size={size}, "
                f"in_degree={in_degree}, "
                f"strategy={mutation_strategy}, "
                f"mutation={mutation_type}, "
                f"repeat={repeat_idx + 1}"
            )

            results = run_rq2_med_experiment_steps(
                size=size,
                interval_width=interval_width,
                width_dist_type=width_dist_type,
                in_degree=in_degree,
                num_samples=num_samples,
                search_method=search_method,
                mutation_type=mutation_type,
                mutation_strategy=mutation_strategy
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

            # all_experiments.append(experiment_obj)
            run_counter += 1

    # return all_experiments


def generate_med_test_lcn():
    lcn = generate_and_save_lcn("medical_data.csv","rq2_med_lcns.json", size=10, in_degree=4)
    print("done")