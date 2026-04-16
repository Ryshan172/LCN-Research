import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch, BIC
import json

"""
Workflow for application of RQ1 flow to MIMIC-IV Dataset

- First has steps for generating LCNs from data
- Then has application of similar workflow found in `rq1_experiments.py`

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



def build_lcn_from_data():
    """
    Full pipeline:
    raw medical data → LCN structure
    """

    # Step 1: Preprocess data
    df = load_medical_csv("medical_data.csv")
    df = preprocess_medical_data(df)
    # Encoding fix
    df = encode_categoricals(df) 

    # Step 2: structure learning
    nodes, edges = learn_structure(df)

    # STEP 3: credal sets
    credal_sets = build_credal_sets(df, nodes, edges)

    # Step 4: logical constraints
    logical_constraints = extract_logical_constraints(df, nodes, edges)

    # LCN output
    lcn = {
        "nodes": nodes,
        "edges": edges,
        "credal_sets": credal_sets,
        "logical_constraints": logical_constraints
    }

    # Export to json
    with open("med_lcn.json", "w") as f:
        json.dump(lcn, f, indent=2)

    print("LCN saved to med_lcn.json")


build_lcn_from_data()