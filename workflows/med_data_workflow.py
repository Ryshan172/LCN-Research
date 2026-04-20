import itertools

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
from workflows.rq1_experiments import build_bn_from_lcn_learned, contingency_sample_lcn, interval_bic_structure_learn, run_interval_lcn_shd, run_structural_hamming_distance, structure_learn_baseline_bn


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


def load_derived_dataset(path):
    """
    Load already-derived dataset (X1..Xn).
    Ensures strict boolean format for BN/LCN pipeline.
    Fixes pandas FutureWarning about silent downcasting.
    """

    import pandas as pd

    df = pd.read_csv(path)

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    for col in df.columns:
        mapped = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "1": True, "0": False,
                "true": True, "false": False
            })
        )

        # Explicit dtype handling (fixes warning)
        mapped = mapped.astype("boolean")   # pandas nullable boolean
        mapped = mapped.fillna(False)

        df[col] = mapped.astype(bool)       # final strict bool

    return df


def topological_sort(nodes, edges):
    from collections import defaultdict, deque

    graph = defaultdict(list)
    indegree = {n: 0 for n in nodes}

    for p, c in edges:
        graph[p].append(c)
        indegree[c] += 1

    q = deque([n for n in nodes if indegree[n] == 0])
    ordered = []

    while q:
        n = q.popleft()
        ordered.append(n)

        for nxt in graph[n]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)

    # fallback for cycles / disconnected nodes
    for n in nodes:
        if n not in ordered:
            ordered.append(n)

    return ordered


def encode_categoricals(df):
    """
    Ensure all variables are boolean-safe and compatible with LCN + pgmpy sampling.
    Prevents silent NaN/str drift that later causes missing-node errors.
    """

    df = df.copy()

    for col in df.columns:

        # already clean bool → keep
        if df[col].dtype == bool:
            continue

        # string-like columns
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({
                    "true": True, "false": False,
                    "1": True, "0": False,
                    "yes": True, "no": False,
                    "m": True, "f": False
                })
            )

        # IMPORTANT FIX:
        # only apply fillna/astype AFTER mapping
        df[col] = df[col].fillna(False).astype(bool)

    return df


def preprocess_medical_data(df):
    """
    Converts raw medical data into stable binary features.
    NO downstream semantic names leak beyond this point.
    """

    df = df.copy()

    # ensure required columns exist
    required_raw = ["anchor_age", "los_hours", "num_diagnoses", "gender", "admission_type"]

    for col in required_raw:
        if col not in df.columns:
            df[col] = 0 if col not in ["gender", "admission_type"] else "unknown"

    # numeric coercion
    df["anchor_age"] = pd.to_numeric(df["anchor_age"], errors="coerce").fillna(0)
    df["los_hours"] = pd.to_numeric(df["los_hours"], errors="coerce").fillna(0)
    df["num_diagnoses"] = pd.to_numeric(df["num_diagnoses"], errors="coerce").fillna(0)

    # engineered binary signals (TEMP internal only)
    features = pd.DataFrame()

    features["f_age_high"] = df["anchor_age"] >= 65
    features["f_los_high"] = df["los_hours"] >= 72
    features["f_diag_high"] = df["num_diagnoses"] >= 10

    features["f_gender_male"] = df["gender"].astype(str).str.upper().eq("M")

    features["f_emergency_admission"] = df["admission_type"].astype(str).isin([
        "URGENT", "EW EMER.", "DIRECT EMER."
    ])

    return features.fillna(False).astype(bool)


def anonymise_columns(df):
    """
    Converts meaningful medical features → anonymous X1..Xn nodes.
    This is what the BN / LCN sees.
    """

    df = df.copy()
    mapping = {}
    anonymised = pd.DataFrame()

    for i, col in enumerate(df.columns):
        x_name = f"X{i+1}"
        anonymised[x_name] = df[col].astype(bool)
        mapping[x_name] = col   # optional debug trace (NOT used downstream)

    return anonymised, mapping


def learn_structure(df, max_parents=1, max_children=2):
    """
    Stable DAG learning with strict schema safety:
    - removes self-loops
    - ensures deterministic ordering
    - enforces max_children correctly
    - guarantees edges reference valid nodes only
    """

    hc = HillClimbSearch(df)

    model = hc.estimate(
        scoring_method=BIC(df),
        max_indegree=max_parents
    )

    nodes = list(df.columns)
    edges = list(model.edges())

    # strict validity filter
    edges = [
        (p, c)
        for p, c in edges
        if p in nodes and c in nodes and p != c
    ]

    # deterministic ordering (VERY important for reproducibility)
    edges = sorted(edges)

    child_count = {n: 0 for n in nodes}
    filtered_edges = []

    for p, c in edges:
        if child_count[p] < max_children:
            filtered_edges.append((p, c))
            child_count[p] += 1

    return nodes, filtered_edges


def filter_edges_by_strength(df, edges, threshold=0.05):
    """
    Remove weak edges based on correlation / dependency strength
    """

    strong_edges = []

    for parent, child in edges:
        try:
            corr = abs(df[parent].corr(df[child]))
        except:
            corr = 0

        if corr >= threshold:
            strong_edges.append((parent, child))

    return strong_edges


def limit_edges(edges, max_edges=4):
    """
    Limit edges but NEVER return empty structure.
    """
    if len(edges) == 0:
        return edges

    return edges[:max_edges]


def build_credal_sets(df, nodes, edges, B=30):

    credal_sets = {}

    for node in nodes:

        if node not in df.columns:
            continue  # hard safety guard

        parents = [p for p, c in edges if c == node]
        credal_sets[node] = {}

        # Root Node
        if len(parents) == 0:

            samples = []
            for _ in range(B):
                boot = df.sample(len(df), replace=True)
                samples.append(boot[node].mean())

            low, high = np.quantile(samples, [0.05, 0.95])

            credal_sets[node]["[]"] = {
                "True": [float(low), float(high)],
                "False": [float(1 - high), float(1 - low)]
            }

        # Conditional Node
        else:

            safe_parents = [p for p in parents if p in df.columns]

            if len(safe_parents) == 0:
                continue

            grouped = df.groupby(safe_parents, dropna=False)

            for config, idx in grouped.groups.items():

                subset = df.loc[idx]
                if len(subset) == 0:
                    continue

                samples = []
                for _ in range(B):
                    boot = subset.sample(len(subset), replace=True)
                    samples.append(boot[node].mean())

                low, high = np.quantile(samples, [0.05, 0.95])

                if not isinstance(config, tuple):
                    config = (config,)

                key = "[" + ", ".join(
                    f"{p}={bool(v)}" for p, v in zip(safe_parents, config)
                ) + "]"

                credal_sets[node][key] = {
                    "True": [float(low), float(high)],
                    "False": [float(1 - high), float(1 - low)]
                }

    return credal_sets


def extract_logical_constraints(df, nodes, edges, threshold=0.85, min_support=10):

    constraints = []

    # Schema alignment (prevents KeyErrors)
    df = df.reindex(columns=nodes, fill_value=False)

    for child in nodes:

        parents = [p for p, c in edges if c == child]
        parents = [p for p in parents if p in df.columns]

        if len(parents) == 0:
            continue

        grouped = df.groupby(parents, dropna=False)[child].agg(["mean", "count"])

        for config, row in grouped.iterrows():

            if row["count"] < min_support:
                continue

            prob = row["mean"]

            if not isinstance(config, tuple):
                config = (config,)

            condition = dict(zip(parents, [bool(v) for v in config]))

            if prob >= threshold:
                constraints.append({"if": condition, "then": {child: True}})

            elif prob <= (1 - threshold):
                constraints.append({"if": condition, "then": {child: False}})

    return constraints


def enforce_schema(df, nodes):
    """
    Guarantees df ALWAYS contains all nodes with strict boolean safety.
    """

    df = df.copy()

    for n in nodes:
        if n not in df.columns:
            df[n] = False

    # drop extra columns + enforce order
    return df[nodes].fillna(False).astype(bool)


def ensure_no_isolated_nodes(nodes, edges):
    """
    Ensures every node appears in at least one edge.
    If isolated, attach it to a random node.
    """
    connected = set()
    for p, c in edges:
        connected.add(p)
        connected.add(c)

    missing = [n for n in nodes if n not in connected]

    for n in missing:
        # connect to a random other node (deterministically safe option: first node)
        target = next(x for x in nodes if x != n)
        edges.append((n, target))

    return edges


def get_variable_node_set(df, min_nodes=4, max_nodes=None, seed=None):
    """
    Randomly selects a subset of X1..Xn columns to define LCN size dynamically.
    """

    if seed is not None:
        np.random.seed(seed)

    cols = list(df.columns)

    if max_nodes is None:
        max_nodes = len(cols)

    k = np.random.randint(min_nodes, max_nodes + 1)
    k = min(k, len(cols))

    selected = np.random.choice(cols, size=k, replace=False).tolist()

    return sorted(selected)


def generate_lcns_from_data(
    csv_path,
    n_lcns=100,
    min_size=4,
    max_size=None,
    max_edges_per_node=2
):
    """
    Generate LCNs of varying sizes with consistent structure + credal sets.
    """

    df_anonymised = load_derived_dataset(csv_path)

    lcns = []

    for _ in range(n_lcns):

        # bootstrap sample
        df_boot = df_anonymised.sample(frac=0.7, replace=True)

        # =========================
        # VARIABLE LCN SIZE HERE
        # =========================
        nodes = get_variable_node_set(
            df_boot,
            min_nodes=min_size,
            max_nodes=max_size or len(df_boot.columns),
        )

        df_boot = df_boot[nodes]
        df_boot = enforce_schema(df_boot, nodes)

        # learn structure on variable node set
        _, learned_edges = learn_structure(df_boot)

        # clean edges
        edges = [
            (p, c)
            for p, c in learned_edges
            if p in nodes and c in nodes and p != c
        ]

        edges = sorted(edges)

        # soft cap instead of fixed 4
        if max_edges_per_node is not None:
            edges = edges[: len(nodes) * max_edges_per_node]

        # fallback if empty
        if len(edges) == 0:
            _, fallback_edges = learn_structure(df_boot)
            edges = [
                (p, c)
                for p, c in fallback_edges
                if p in nodes and c in nodes and p != c
            ]

        # =========================
        # ensure connectivity
        # =========================
        connected = set()
        for p, c in edges:
            connected.add(p)
            connected.add(c)

        for n in nodes:
            if n not in connected:
                target = next(x for x in nodes if x != n)
                edges.append((n, target))

        # topological ordering
        nodes = topological_sort(nodes, edges)

        # parent map
        parent_map = {n: [] for n in nodes}
        for p, c in edges:
            parent_map[c].append(p)

        for n in nodes:
            parent_map[n] = sorted(parent_map[n])

        # =========================
        # credal sets (still uniform placeholder)
        # =========================
        credal_sets = {}

        for node in nodes:
            parents = parent_map[node]
            credal_sets[node] = {}

            if len(parents) == 0:
                credal_sets[node]["[]"] = {
                    "True": [0.5, 0.5],
                    "False": [0.5, 0.5]
                }
            else:
                for combo in itertools.product([False, True], repeat=len(parents)):

                    key = "[" + ", ".join(
                        f"{p}={v}" for p, v in zip(parents, combo)
                    ) + "]"

                    credal_sets[node][key] = {
                        "True": [0.5, 0.5],
                        "False": [0.5, 0.5]
                    }

        lcns.append({
            "nodes": nodes,
            "edges": edges,
            "credal_sets": credal_sets,
            "logical_constraints": []
        })

    return lcns


# Running worflow on a single medical data LCN
def run_workflow_on_given_lcn(lcn, num_samples):

    model, sampled_states = build_precise_bn_from_lcn(lcn)

    expected_nodes = lcn["nodes"]

    if len(expected_nodes) == 0:
        raise ValueError("LCN has no nodes")

    # Sample from LCN
    lcn_aggregate_table, lcn_samples_df = contingency_sample_lcn(lcn, num_samples)

    lcn_samples_df = enforce_schema(lcn_samples_df, expected_nodes)
    lcn_samples_df = lcn_samples_df[expected_nodes].astype(bool)

    # Baseline BN sampling
    bn_forward_samples = ancestral_sample_bn(model, n_samples=num_samples)

    # Learn baseline BN structure
    learned_bn = structure_learn_baseline_bn(bn_forward_samples)

    # Build learned BN model (for KL)
    edges = list(learned_bn["hillclimb_edges"])

    learned_bn_model = DiscreteBayesianNetwork(edges)
    learned_bn_model.add_nodes_from(model.nodes())
    learned_bn_model.fit(bn_forward_samples)


    # SHD (baseline structure)
    baseline_shd_results = run_structural_hamming_distance(model, learned_bn)


    # Interval BIC learning
    interval_bic_results = interval_bic_structure_learn(
        lcn_aggregate_table,
        lcn_samples_df
    )

    interval_lcn_shd_results = run_interval_lcn_shd(
        model,
        interval_bic_results
    )

    # Build LCN-learned BN model (mid)
    lcn_learned_bn = build_bn_from_lcn_learned(
        interval_bic_results,
        model,
        bn_forward_samples
    )


    # KL: Baseline BN vs Learned BN
    kl_baseline_vs_learned = kl_divergence_from_samples(
        true_model=model,
        approx_model=learned_bn_model,
        samples_df=bn_forward_samples
    )

    # KL: Baseline BN vs LCN-learned BN
    kl_baseline_vs_lcn = kl_divergence_from_samples(
        true_model=model,
        approx_model=lcn_learned_bn,
        samples_df=bn_forward_samples
    )

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
        },
        "kl_divergence": {
            "baseline_vs_learned_bn": kl_baseline_vs_learned,
            "baseline_vs_lcn_learned_bn": kl_baseline_vs_lcn
        },

        # Keep models for debugging / analysis
        "learned_bn_model": learned_bn_model,
        "lcn_learned_bn_model": lcn_learned_bn
    }

def run_medical_experiments(csv_path, n_lcns=100, num_samples=300):
    """
    Run experiments workflow similar to RQ1 but on medical data 
    - Uses a fixed number of samples
    """

    lcns = generate_lcns_from_data(csv_path, n_lcns)
    

    for i, lcn in enumerate(lcns):
        
        print("LCN:")
        print(lcn)

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

        save_application_to_json(experiment_obj, f"run_{i+1}")