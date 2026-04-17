import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch, BIC

# Function Imports
from metric_functions.kl_divergence import kl_divergence_from_samples
from metric_functions.structural_hamming_distance import structural_hamming_distance_compare
from sampler_functions.bn_topological import ancestral_sample_bn, build_precise_bn_from_lcn
from sampler_functions.contingency_sampler import credal_aggregate_intervals, sample_dataset
from structure_learning.hill_climbing import run_hillclimbing_bic, run_interval_bic_hillclimb
from scoring_functions.interval_bic_derivation import compute_interval_BIC, compute_network_interval_BIC
from utils.data_saving import save_application_to_json, save_experiment_to_json
from workflows.rq1_experiments import contingency_sample_lcn, interval_bic_structure_learn, run_interval_lcn_shd, run_structural_hamming_distance, structure_learn_baseline_bn


"""
Workflow similar to RQ1 but for medical data application
"""

def load_medical_csv(path):
    """
    Load raw medical CSV into a pandas DataFrame.

    This ensures:
    - correct parsing
    - stable data types
    - consistent preprocessing downstream
    """

    df = pd.read_csv(path)

    # Basic cleanup to remove accidental spaces
    df.columns = [c.strip() for c in df.columns]

    # Type enforcement (important for BN learning)
    numeric_cols = [
        "anchor_age",
        "los_hours",
        "num_diagnoses"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


    # Fill missing values safely
    df = df.fillna({
        "gender": "unknown",
        "admission_type": "unknown"
    })

    return df


def encode_categoricals(df):
    """
    Convert categorical/string columns into integer codes
    so numeric operations like mean() work.
    """

    for col in df.columns:
        if df[col].dtype == "object" or pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("category").cat.codes

    return df


def preprocess_medical_data(df):
    """
    Converts raw clinical table into discrete variables for BN/LCN learning.
    """

    df = df.copy()

    # Age -> categorical bins
    df["age_group"] = pd.cut(
        df["anchor_age"],
        bins=[0, 50, 70, 120],
        labels=["young", "middle", "elderly"]
    )

    # Length of stay -> bins
    df["los_group"] = pd.cut(
        df["los_hours"],
        bins=[0, 48, 120, float("inf")],
        labels=["short", "medium", "long"]
    )

    # Dropping raw continuous columns and replacing with bins
    df = df.drop(columns=["anchor_age", "los_hours", "hadm_id"])

    # Ensure categorical encoding consistency
    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].cat.add_categories(["unknown"])
            df[col] = df[col].fillna("unknown")
        elif df[col].dtype == "object":
            df[col] = df[col].fillna("unknown")
        else:
            df[col] = df[col].fillna(0)

    return df


def learn_structure(df, max_parents=2):
    """
    Learns a DAG structure from data using Hill Climbing + BIC score.
    """

    hc = HillClimbSearch(df)

    # Learn best-scoring DAG under BIC
    model = hc.estimate(
        scoring_method=BIC(df),
        max_indegree=max_parents
    )

    # Convert to edge list format (your LCN format)
    edges = list(model.edges())

    # Extract nodes
    nodes = list(df.columns)

    return nodes, edges


def build_credal_sets(df, nodes, edges, B=200):
    """
    Builds LCN credal sets using bootstrap uncertainty estimation.
    """

    credal_sets = {}

    for node in nodes:

        # Find parents of node
        parents = [p for p, c in edges if c == node]

        credal_sets[node] = {}

        # Case: Root Node (no parents)
        if len(parents) == 0:
            samples = []

            for _ in range(B):
                boot = df.sample(len(df), replace=True)
                p_true = (boot[node] == 1).mean()
                samples.append(p_true)

            low, high = np.quantile(samples, [0.05, 0.95])

            credal_sets[node]["[]"] = {
                "True": [round(low, 2), round(high, 2)],
                "False": [round(1 - high, 2), round(1 - low, 2)]
            }

        # Case: Conditional Node
        else:
            grouped = df.groupby(parents)

            for config, subset_idx in grouped.groups.items():

                subset = df.loc[subset_idx]

                if len(subset) == 0:
                    continue

                samples = []

                for _ in range(B):
                    boot = subset.sample(len(subset), replace=True)
                    p_true = (boot[node] == 1).mean()
                    samples.append(p_true)

                low, high = np.quantile(samples, [0.05, 0.95])

                # Build readable parent condition key
                if isinstance(config, tuple):
                    key = "[" + ", ".join(
                        f"{p}={v}" for p, v in zip(parents, config)
                    ) + "]"
                else:
                    key = f"[{parents[0]}={config}]"

                credal_sets[node][key] = {
                    "True": [round(low, 2), round(high, 2)],
                    "False": [round(1 - high, 2), round(1 - low, 2)]
                }

    return credal_sets


def extract_logical_constraints(df, nodes, edges, threshold=0.95):
    """
    Extracts deterministic rules from data:
    IF parent config → child is almost always True/False
    """

    constraints = []

    for child in nodes:

        parents = [p for p, c in edges if c == child]

        if len(parents) == 0:
            continue

        grouped = df.groupby(parents)[child].mean()

        for config, prob in grouped.items():

            # Convert config to dict
            if not isinstance(config, tuple):
                config = (config,)

            condition = dict(zip(parents, config))

            # High certainty rule
            if prob >= threshold:
                constraints.append({
                    "if": condition,
                    "then": {child: True}
                })

            # Low certainty rule
            elif prob <= (1 - threshold):
                constraints.append({
                    "if": condition,
                    "then": {child: False}
                })

    return constraints


# Generate LCNs from medical data bootstrap
def generate_lcns_from_data(csv_path, n_lcns=100):

    df_full = load_medical_csv(csv_path)
    df_full = preprocess_medical_data(df_full)
    df_full = encode_categoricals(df_full)

    lcns = []

    for i in range(n_lcns):
        print(f"Generating LCN {i+1}/{n_lcns}")

        df_boot = df_full.sample(len(df_full), replace=True)

        nodes, edges = learn_structure(df_boot)
        credal_sets = build_credal_sets(df_boot, nodes, edges)
        logical_constraints = extract_logical_constraints(df_boot, nodes, edges)

        lcn = {
            "nodes": nodes,
            "edges": edges,
            "credal_sets": credal_sets,
            "logical_constraints": logical_constraints
        }

        lcns.append(lcn)

    return lcns



# Running worflow on a single medical data LCN
def run_workflow_on_given_lcn(lcn, num_samples):

    model, sampled_states = build_precise_bn_from_lcn(lcn)

    lcn_aggregate_table, lcn_samples_df = contingency_sample_lcn(lcn, num_samples)

    bn_forward_samples = ancestral_sample_bn(model, n_samples=num_samples)

    learned_bn = structure_learn_baseline_bn(bn_forward_samples)

    baseline_shd_results = run_structural_hamming_distance(model, learned_bn)

    interval_bic_results = interval_bic_structure_learn(
        lcn_aggregate_table,
        lcn_samples_df
    )

    interval_lcn_shd_results = run_interval_lcn_shd(model, interval_bic_results)

    return {
        "lcn": lcn,
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
        }
    }


def run_medical_experiments(csv_path, n_lcns=100, num_samples=300):

    lcns = generate_lcns_from_data(csv_path, n_lcns)

    for i, lcn in enumerate(lcns):

        print(f"\nRunning experiment {i+1}/{n_lcns}")

        results = run_workflow_on_given_lcn(lcn, num_samples)

        experiment_obj = {
            "run_id": i + 1,
            "repeat": 1,
            "params": {
                "num_samples": num_samples
            },
            "results": results
        }

        save_application_to_json(experiment_obj, f"medical_run_{i+1}")


# run_medical_experiments(
#     csv_path="medical_data.csv",
#     n_lcns=10,
#     num_samples=300
# )