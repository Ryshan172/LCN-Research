
import itertools

import pandas as pd


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df.astype(int)


import itertools
import numpy as np


def generate_basic_lcn(csv_path, corr_threshold=0.15, max_parents=2):
    """
    Creates a realistic initial LCN with controlled sparsity.

    Key properties:
    - weak data-driven structure (not learned)
    - bounded parent count (prevents exponential credal growth)
    - supports BIC / IBIC optimisation fairly
    """

    df = load_dataset(csv_path)

    nodes = list(df.columns)

    edges = []

    # Track how many parents each node already has
    # This prevents dense CPT explosion later
    parent_count = {n: 0 for n in nodes}

    # Build weak structure using correlation
    # This is ONLY a heuristic seed, not learning
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:

            corr = np.corrcoef(df[a], df[b])[0, 1]

            if np.abs(corr) > corr_threshold:

                # enforce max parents constraint (critical fix)
                if parent_count[b] < max_parents:
                    edges.append((a, b))
                    parent_count[b] += 1

    # Build parent map from edges
    parent_map = {n: [] for n in nodes}

    for p, c in edges:
        parent_map[c].append(p)

    # Credal sets (uncertainty model)
    credal_sets = {}

    for node in nodes:

        parents = parent_map[node]
        credal_sets[node] = {}

        # Root nodes: use empirical uncertainty
        if len(parents) == 0:

            p_true = df[node].mean()
            eps = 0.1

            credal_sets[node]["[]"] = {
                "True": [
                    max(0.01, p_true - eps),
                    min(0.99, p_true + eps)
                ],
                "False": [
                    max(0.01, (1 - p_true) - eps),
                    min(0.99, (1 - p_true) + eps)
                ]
            }

        # Conditional nodes (kept simple at init stage)
        else:

            # IMPORTANT DESIGN CHOICE:
            # We do NOT fully expand CPTs at init to avoid combinatorial explosion.
            #
            # Instead:
            # - keep placeholder credal sets
            # - let optimisation refine structure first
            for combo in itertools.product([False, True], repeat=len(parents)):

                key = "[" + ", ".join(
                    f"{p}={v}" for p, v in zip(parents, combo)
                ) + "]"

                credal_sets[node][key] = {
                    "True": [0.5, 0.5],
                    "False": [0.5, 0.5]
                }

    logical_constraints = []

    return {
        "nodes": nodes,
        "edges": edges,
        "credal_sets": credal_sets,
        "logical_constraints": logical_constraints
    }