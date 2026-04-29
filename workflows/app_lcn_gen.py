
import itertools
import pandas as pd
import numpy as np
import copy
import random
from collections import defaultdict

from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.estimators import BIC as BicScore

from lcn_functions.lcn_check import validate_generated_lcn
from scoring_functions.interval_bic_derivation import compute_interval_BIC, compute_network_interval_BIC
from workflows.rq1_experiments import contingency_sample_lcn, interval_bic_structure_learn


# ----------------- Step 1 code ---------------------------------------------------

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df.astype(int)


def generate_basic_lcn(csv_path, corr_threshold=0.15, max_parents=2):
    """
    Creates a realistic initial LCN with controlled sparsity.

    Key properties:
    - weak data-driven structure (not learned)
    - bounded parent count (prevents exponential credal growth)
    - supports BIC / IBIC optimisation fairly
    """

    # Return this as well as the result for later use
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

    lcn =  {
        "nodes": nodes,
        "edges": edges,
        "credal_sets": credal_sets,
        "logical_constraints": logical_constraints
    }

    return lcn, df


# ----------------------- Step 2 Code ----------------------------

def compute_bic_score(df, edges):
    """
    Computes BIC score for a given graph structure.

    - BIC is used ONLY for structure comparison (not LCN semantics)
    - It evaluates how well a DAG explains observed binary data
    - Ignores credal sets completely (important design separation)

    NOTE:
    - This assumes a standard Bayesian Network interpretation
    - LCN uncertainty layer is NOT included here
    """

    model = BayesianNetwork(edges)

    try:
        score = BicScore(df).score(model)
        return score
    except Exception:
        return -np.inf
    

def mutate_edges(nodes, edges):
    """
    Generates a neighbouring graph via a single random mutation.

    WHY THIS EXISTS:
    - Hill Climbing requires exploring "nearby" graph structures
    - Mutations define the search space:
        * add edge
        * remove edge
        * reverse edge
    """

    edges = copy.deepcopy(edges)

    operation = random.choice(["add", "remove", "reverse"])

    if operation == "add":
        a, b = random.sample(nodes, 2)
        new_edge = (a, b)

        if new_edge not in edges:
            edges.append(new_edge)

    elif operation == "remove" and edges:
        edges.pop(random.randint(0, len(edges) - 1))

    elif operation == "reverse" and edges:
        i = random.randint(0, len(edges) - 1)
        a, b = edges[i]
        edges[i] = (b, a)

    return edges

def is_valid_dag(nodes, edges, max_parents=2):
    """
    Ensures a candidate graph is structurally valid.

    WHY THIS EXISTS:
    - LCN requires bounded complexity (max parents constraint)
    - BIC scoring requires DAGs (no cycles)
    - Prevents exponential CPT explosion later
    """

    # --- parent constraint ---
    parent_count = {n: 0 for n in nodes}

    for a, b in edges:
        if a == b:
            return False

        parent_count[b] += 1
        if parent_count[b] > max_parents:
            return False

    # --- cycle detection ---
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)

    visited = set()
    stack = set()

    def dfs(node):
        if node in stack:
            return False
        if node in visited:
            return True

        stack.add(node)

        for nxt in graph[node]:
            if not dfs(nxt):
                return False

        stack.remove(node)
        visited.add(node)
        return True

    return all(dfs(n) for n in nodes)


def optimize_lcn_bic(initial_lcn, df, max_iters=100, max_parents=2):
    """
    Greedy Hill Climbing using BIC with full LCN validation.

    WHY THIS VERSION EXISTS:
    - ensures all candidate graphs remain structurally valid LCNs
    - prevents invalid CPT explosions from entering scoring stage
    - uses external validation function as a hard constraint gate
    """

    nodes = initial_lcn["nodes"]
    best_edges = initial_lcn["edges"]

    best_lcn = initial_lcn
    best_score = compute_bic_score(df, best_edges)

    history = [best_score]

    for _ in range(max_iters):

        # Step 1: propose mutation
        candidate_edges = mutate_edges(nodes, best_edges)

        candidate_lcn = {
            "nodes": nodes,
            "edges": candidate_edges,
            "credal_sets": initial_lcn["credal_sets"],
            "logical_constraints": initial_lcn["logical_constraints"]
        }

        # Step 2: FULL LCN VALIDATION
        if not validate_generated_lcn(candidate_lcn):
            continue

        # Step 3: enforce DAG + structural constraints (extra safety)
        if not is_valid_dag(nodes, candidate_edges, max_parents):
            continue

        # Step 4: score candidate
        score = compute_bic_score(df, candidate_edges)

        # Step 5: greedy acceptance
        if score > best_score or random.random() < 0.05:
            best_score = score
            best_edges = candidate_edges
            best_lcn = candidate_lcn
            history.append(best_score)

    return {
        "nodes": nodes,
        "edges": best_edges,
        "bic_score": best_score,
        "history": history,
        "credal_sets": best_lcn["credal_sets"],
        "logical_constraints": best_lcn["logical_constraints"]
    }


def optimize_lcn_ibic(initial_lcn, df, max_iters=100, max_parents=2):
    """
    Greedy Hill Climbing using IBIC instead of BIC.

    IMPORTANT DESIGN POINT:
    - search procedure is identical to BIC version
    - ONLY scoring function changes
    - ensures fair comparison between BIC vs IBIC structure learning
    """

    nodes = initial_lcn["nodes"]
    best_edges = initial_lcn["edges"]

    best_lcn = initial_lcn

    # ---- IBIC setup step ----
    # To computer the IBIC, first you need the aggregate table and samples
    lcn_aggregate_table, lcn_samples_df = contingency_sample_lcn(initial_lcn, 200)

    ibic_results = interval_bic_structure_learn(
        lcn_aggregate_table,
        lcn_samples_df
    )

    best_score = ibic_results["network_interval"][1]  # upper bound as default
    history = [best_score]

    for _ in range(max_iters):

        # 1. mutation (UNCHANGED)
        candidate_edges = mutate_edges(nodes, best_edges)

        candidate_lcn = {
            "nodes": nodes,
            "edges": candidate_edges,
            "credal_sets": initial_lcn["credal_sets"],
            "logical_constraints": initial_lcn["logical_constraints"]
        }

        # 2. validation (UNCHANGED)
        if not validate_generated_lcn(candidate_lcn):
            continue

        if not is_valid_dag(nodes, candidate_edges, max_parents):
            continue

        # 3. IBIC scoring (NEW)
        candidate_table, candidate_samples = contingency_sample_lcn(candidate_lcn, 500)

        interval_per_node = compute_interval_BIC(candidate_table)
        network_interval = compute_network_interval_BIC(interval_per_node)

        candidate_score = (network_interval[0] + network_interval[1]) / 2  # MID

        # 4. greedy update
        if candidate_score > best_score:
            best_score = candidate_score
            best_edges = candidate_edges
            best_lcn = candidate_lcn
            history.append(best_score)

    return {
        "nodes": nodes,
        "edges": best_edges,
        "ibic_score": best_score,
        "history": history,
        "credal_sets": best_lcn["credal_sets"],
        "logical_constraints": best_lcn["logical_constraints"]
    }