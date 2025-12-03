import json
import random
from collections import defaultdict, deque
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import itertools
import numpy as np


"""
Algorithm which samples a single precise Bayesian Network (BN) from a Logical
Credal Network (LCN). Nodes are processed in topological order so that
all parents are handled before their children. 

For each node:

1. Determines which parent configuration actually applies based on the already
   sampled parent states.

2. Checks whether any logical constraints force the node to a deterministic
   value. If so, it creates a deterministic CPT entry (prob = 0 or 1).

3. If no rule forces the outcome, it samples probabilities from the interval
   bounds in the node’s credal set (e.g., p ∈ [l, u]).

4. It then samples an actual boolean state using the chosen probability so that
   descendants can use it when picking their valid credal rows.

Only the reachable parent configuration is used; all others are ignored.

The resulting BN:
   • Respects the causal structure,
   • Obeys all logical constraints,
   • Chooses probabilities within the LCN’s interval bounds
"""


# Topological ordering
def topological_sort(nodes, edges):
    # Compute indegree (number of parents) for each node
    indegree = {n: 0 for n in nodes}
    graph = defaultdict(list)

    # Build adjacency list and track indegrees
    for p, c in edges:
        graph[p].append(c)
        indegree[c] += 1

    # Start with all nodes that have no parents
    queue = deque([n for n in nodes if indegree[n] == 0])
    order = []

    # Kahn’s algorithm: repeatedly remove parentless nodes
    while queue:
        node = queue.popleft()
        order.append(node)

        # "Removing" node reduces indegree of its children
        for child in graph[node]:
            indegree[child] -= 1
            # If child has no remaining parents, process it next
            if indegree[child] == 0:
                queue.append(child)

    # Result is a parent-before-child ordering
    return order


# Parse "[X1=True, X3=False]" into a dict
def parse_parent_config_string(pc_string):
    # Handle root configuration: no parents
    if pc_string == "[]":
        return {}

    # Remove brackets and split into individual assignments
    pc_string = pc_string.strip("[]")
    assignments = pc_string.split(",")
    config = {}

    # Convert "X1=True" into {"X1": True}
    for a in assignments:
        var, val = a.split("=")
        config[var.strip()] = (val.strip() == "True")

    return config


# Check if any logical constraints force this node to a deterministic value
def apply_logical_constraints(node, parent_config, constraints):
    forced_value = None

    # Iterate over all constraints
    for rule in constraints:
        if_cond = rule["if"]     # condition part of the rule
        then_cond = rule["then"] # consequence part of the rule

        # Check if the IF condition matches the current parent configuration
        satisfied = True
        for var, val in if_cond.items():
            if parent_config.get(var) != val:
                satisfied = False
                break

        # If condition is satisfied and the rule applies to this node, force its value
        if satisfied and node in then_cond:
            forced_value = then_cond[node]

    # Returns True/False if forced, otherwise None
    return forced_value


# interval sampling
def sample_from_interval(interval):
    l, u = interval
    if l == u:
        return l
    return random.uniform(l, u)


# Build one precise BN sampled from the LCN
def build_precise_bn_from_lcn(lcn):
    """
    Sample a single precise Bayesian Network from a Logical Credal Network (LCN)
    using topological order. Returns a pgmpy DiscreteBayesianNetwork and sampled states.
    """

    nodes = lcn["nodes"]
    edges = lcn["edges"]
    credal_sets = lcn["credal_sets"]
    constraints = lcn.get("logical_constraints", [])

    # Get parents of each node
    parents = defaultdict(list)
    for p, c in edges:
        parents[c].append(p)

    # Topological order
    topo_order = topological_sort(nodes, edges)

    # Store sampled states for propagation
    sampled_states = {}

    # Initialize DiscreteBayesianNetwork
    model = DiscreteBayesianNetwork(edges)

    # Build CPTs
    for node in topo_order:
        parent_list = parents[node]

        # Current parent configuration based on sampled states
        pc = {p: sampled_states[p] for p in parent_list}
        pc_string = "[]" if not pc else "[" + ", ".join(f"{k}={v}" for k, v in pc.items()) + "]"

        # Apply logical constraints
        forced_value = apply_logical_constraints(node, pc, constraints)

        if forced_value is not None:
            # Deterministic CPT
            values = [[1.0 if forced_value else 0.0],
                      [0.0 if forced_value else 1.0]]
        else:
            # Sample from credal interval
            credal_row = credal_sets[node][pc_string]
            p_true = sample_from_interval(credal_row["True"])
            p_false = 1.0 - p_true
            values = [[p_true], [p_false]]

        # Sample actual state for propagation
        sampled_states[node] = random.random() < values[0][0]

        # Create TabularCPD
        if not parent_list:
            cpd = TabularCPD(variable=node, variable_card=2, values=values)
        else:
            n_combinations = 2 ** len(parent_list)
            full_values = [[0]*n_combinations, [0]*n_combinations]

            # Find index of sampled parent combination
            idx = sum(2**(len(parent_list)-i-1) if pc[p] else 0 for i, p in enumerate(parent_list))
            full_values[0][idx] = values[0][0]
            full_values[1][idx] = values[1][0]

            parent_card = [2] * len(parent_list)
            cpd = TabularCPD(variable=node, variable_card=2, values=full_values,
                             evidence=parent_list, evidence_card=parent_card)

        # Add CPT to model
        model.add_cpds(cpd)

    return model, sampled_states


def bn_to_json(model: DiscreteBayesianNetwork):
    """
    Convert a DiscreteBayesianNetwork to a JSON-serializable dict,
    including nodes, edges, and CPTs. Ensures all NumPy types are converted
    to native Python types.
    """
    # Convert NodeView and EdgeView to lists
    nodes = list(model.nodes())
    edges = list(model.edges())

    # Helper function to recursively convert NumPy types to Python
    def convert_to_python(obj):
        if isinstance(obj, (np.ndarray, list, tuple)):
            return [convert_to_python(o) for o in obj]
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        else:
            return obj

    # Get CPTs as a dictionary
    cpts = {}
    for cpd in model.get_cpds():
        cpts[cpd.variable] = {
            "values": convert_to_python(cpd.get_values()),
            "evidence": list(cpd.variables[1:]) if len(cpd.variables) > 1 else [],
            "evidence_card": convert_to_python(list(cpd.cardinality[1:])) if len(cpd.cardinality) > 1 else []
        }

    return {
        "nodes": nodes,
        "edges": edges,
        "cpts": cpts
    }
