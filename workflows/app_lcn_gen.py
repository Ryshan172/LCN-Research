
import itertools
import pandas as pd
import numpy as np
import copy
import random
from collections import defaultdict

from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.estimators import BIC as BicScore

from lcn_functions.lcn_check import validate_generated_lcn
from scoring_functions.updated_ibic import RQ1FamilyScores
from structure_learning.hill_climbing import _midpoint_rank
import networkx as nx


# ----------------- Step 1 code ---------------------------------------------------

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    # Change done: reject invalid observations before integer conversion can truncate them.
    if df.empty or not df.isin([0, 1]).all().all():
        raise ValueError("The medical application requires complete binary observations.")
    return df.astype(int)


def generate_basic_lcn(csv_path, corr_threshold=0.15, max_parents=2):
    df = load_dataset(csv_path)
    nodes = list(df.columns)

    edges = []
    parent_count = {n: 0 for n in nodes}

    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            corr = np.corrcoef(df[a], df[b])[0, 1]

            if np.abs(corr) > corr_threshold:
                if parent_count[b] < max_parents:
                    edges.append((a, b))
                    parent_count[b] += 1

    # parent map (not strictly used later, but kept consistent)
    parent_map = {n: [] for n in nodes}
    for p, c in edges:
        parent_map[c].append(p)

    credal_sets = {}

    for node in nodes:
        parents = sorted(parent_map[node])  # IMPORTANT FIX

        credal_sets[node] = {}

        if len(parents) == 0:
            p_true = df[node].mean()
            eps = 0.1

            credal_sets[node]["[]"] = {
                "True": [max(0.01, p_true - eps), min(0.99, p_true + eps)],
                "False": [max(0.01, (1 - p_true) - eps), min(0.99, (1 - p_true) + eps)]
            }
        else:
            for combo in itertools.product([False, True], repeat=len(parents)):
                key = "[" + ", ".join(f"{p}={v}" for p, v in zip(parents, combo)) + "]"

                credal_sets[node][key] = {
                    "True": [0.5, 0.5],
                    "False": [0.5, 0.5]
                }

    return {
        "nodes": nodes,
        "edges": edges,
        "credal_sets": credal_sets,
        "logical_constraints": []
    }, df


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

def is_cyclic(nodes, edges):
    """
    Lightweight cycle check (DFS).
    Returns True if cycle exists.
    """

    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)

    visited = set()
    stack = set()

    def dfs(node):
        if node in stack:
            return True
        if node in visited:
            return False

        stack.add(node)

        for nxt in graph[node]:
            if dfs(nxt):
                return True

        stack.remove(node)
        visited.add(node)
        return False

    return any(dfs(n) for n in nodes)


def mutate_edges(nodes, edges, max_attempts=10):
    edges = copy.deepcopy(edges)

    for _ in range(max_attempts):
        operation = random.choice(["add", "remove", "reverse"])
        candidate = copy.deepcopy(edges)

        if operation == "add":
            a, b = random.sample(nodes, 2)

            if a == b:
                continue

            if (a, b) in candidate:
                continue

            candidate.append((a, b))

        elif operation == "remove" and candidate:
            candidate.pop(random.randint(0, len(candidate) - 1))

        elif operation == "reverse" and candidate:
            i = random.randint(0, len(candidate) - 1)
            a, b = candidate[i]

            if a != b:
                candidate[i] = (b, a)

        if not is_cyclic(nodes, candidate):
            return candidate

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


def sync_credal_sets_with_structure(lcn):
    nodes = [str(n).strip() for n in lcn["nodes"]]
    edges = [(str(a).strip(), str(b).strip()) for a, b in lcn["edges"]]

    parent_map = {n: [] for n in nodes}

    for p, c in edges:
        if c in parent_map:
            parent_map[c].append(p)

    new_credal = {}

    for node in nodes:

        parents = sorted(parent_map.get(node, []))  # MUST match sampler

        new_credal[node] = {}

        # ---------------- ROOT NODE ----------------
        if len(parents) == 0:
            new_credal[node]["[]"] = {
                "True": [0.5, 0.5],
                "False": [0.5, 0.5]
            }
            continue

        # ---------------- FULL CPT COVERAGE ----------------
        # IMPORTANT: must include ALL combinations even if unused
        for combo in itertools.product([False, True], repeat=len(parents)):

            key = "[" + ", ".join(
                f"{p}={'True' if v else 'False'}"
                for p, v in zip(parents, combo)
            ) + "]"

            new_credal[node][key] = {
                "True": [0.5, 0.5],
                "False": [0.5, 0.5]
            }

        # safety fallback (prevents missing-node crash)
        if len(new_credal[node]) == 0:
            new_credal[node]["[]"] = {
                "True": [0.5, 0.5],
                "False": [0.5, 0.5]
            }

    # final safety: ensure ALL nodes exist
    for n in nodes:
        if n not in new_credal:
            new_credal[n] = {
                "[]": {"True": [0.5, 0.5], "False": [0.5, 0.5]}
            }

    lcn["nodes"] = nodes
    lcn["edges"] = edges
    lcn["credal_sets"] = new_credal

    return lcn


def estimate_credal_sets_from_data(df, lcn, eps_scale=0.15, min_eps=0.02):
    """
    Converts structure-defined CPT skeleton into data-driven credal intervals.

    - Uses empirical conditional probabilities
    - Builds interval uncertainty based on sample size
    - Works after sync_credal_sets_with_structure()
    """

    nodes = lcn["nodes"]
    edges = lcn["edges"]

    parent_map = {n: [] for n in nodes}
    for p, c in edges:
        parent_map[c].append(p)

    new_credal = {}

    for node in nodes:
        parents = sorted(parent_map[node])
        new_credal[node] = {}

        # ---------------- ROOT NODE ----------------
        if len(parents) == 0:
            p = df[node].mean()

            # uncertainty based on variance + data size
            eps = max(min_eps, eps_scale * np.std(df[node]))

            new_credal[node]["[]"] = {
                "True": [max(0.01, p - eps), min(0.99, p + eps)],
                "False": [max(0.01, (1 - p) - eps), min(0.99, (1 - p) + eps)]
            }
            continue

        # ---------------- CONDITIONAL NODES ----------------
        for combo in itertools.product([False, True], repeat=len(parents)):

            key = "[" + ", ".join(
                f"{p}={'True' if v else 'False'}"
                for p, v in zip(parents, combo)
            ) + "]"

            # build mask for rows matching parent configuration
            mask = np.ones(len(df), dtype=bool)

            for p, v in zip(parents, combo):
                mask &= (df[p] == (1 if v else 0))

            subset = df[mask]

            # if no data, fall back to weak prior
            if len(subset) == 0:
                p = 0.5
                eps = 0.2
            else:
                p = subset[node].mean()

                # uncertainty shrinks with data size
                eps = max(min_eps, eps_scale / np.sqrt(len(subset)))

            new_credal[node][key] = {
                "True": [max(0.01, p - eps), min(0.99, p + eps)],
                "False": [max(0.01, (1 - p) - eps), min(0.99, (1 - p) + eps)]
            }

    lcn["credal_sets"] = new_credal
    return lcn


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

        candidate_lcn = sync_credal_sets_with_structure(candidate_lcn)

        # Recompute credal sets
        candidate_lcn = estimate_credal_sets_from_data(df, candidate_lcn)

        # Step 2: FULL LCN VALIDATION
        if not validate_generated_lcn(candidate_lcn):
            continue

        # # Step 3: enforce DAG + structural constraints (extra safety)
        # if not is_valid_dag(nodes, candidate_edges, max_parents):
        #     continue

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


def optimize_lcn_ibic(initial_lcn, df, max_iters=100, max_parents=2, *, interval_width=0.2):
    """Select a midpoint IBIC structure using the medical workflow's candidate mutations."""
    nodes = initial_lcn["nodes"]
    rules = initial_lcn["logical_constraints"]
    # Change done: score every mutation on the same observed records using RQ1's
    # propagated full width. Previously each graph generated a different sample.
    families = RQ1FamilyScores(df[nodes], rules, interval_width=interval_width)
    best = nx.DiGraph()
    best.add_nodes_from(nodes)
    best.add_edges_from(initial_lcn["edges"])
    best.add_edges_from(families.required_edges)
    if not is_valid_dag(nodes, best.edges(), max_parents):
        raise ValueError("Initial medical graph violates the DAG or parent limit")
    best_rank = _midpoint_rank(best, families)
    if best_rank == (-np.inf, -np.inf):
        raise ValueError("Medical observations conflict with the supplied logical rules")
    history = [best_rank[0]]
    for _ in range(max_iters):
        edges = mutate_edges(nodes, list(best.edges()))
        if not families.required_edges.issubset(edges) or not is_valid_dag(nodes, edges, max_parents):
            continue
        candidate = nx.DiGraph()
        candidate.add_nodes_from(nodes)
        candidate.add_edges_from(edges)
        # Change done: compare actual midpoints first. On a -inf plateau an
        # improving upper endpoint can distinguish two otherwise tied structures.
        rank = _midpoint_rank(candidate, families)
        if rank > best_rank:
            if rank[0] == best_rank[0] == -np.inf and rank[1] <= best_rank[1] + 1e-10:
                continue
            best, best_rank = candidate, rank
            history.append(rank[0])
    return {
        "nodes": nodes, "edges": list(best.edges()), "ibic_score": best_rank[0],
        "history": history, "logical_constraints": rules,
        "credal_sets": {node: families.fitted_cpt(node, best.predecessors(node)) for node in nodes},
    }
