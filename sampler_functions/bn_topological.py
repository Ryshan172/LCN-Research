import json
import random
from collections import defaultdict, deque
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import itertools
import numpy as np
from pgmpy.sampling import BayesianModelSampling


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

    FIXED VERSION:
    - Each parent configuration gets its own CPT column.
    - Logical constraints applied per parent configuration.
    - Credal intervals sampled independently per configuration.
    """

    nodes = lcn["nodes"]
    edges = lcn["edges"]
    credal_sets = lcn["credal_sets"]
    constraints = lcn.get("logical_constraints", [])

    # Build parent lists
    parents = defaultdict(list)
    for p, c in edges:
        parents[c].append(p)

    # Compute topological order
    topo_order = topological_sort(nodes, edges)

    # Store sampled world state
    sampled_states = {}

    # Build BN model
    model = DiscreteBayesianNetwork(edges)

    # Build CPTs in topological order
    for node in topo_order:
        parent_list = parents[node]

        # Case 1: Node has NO parents
        if not parent_list:
            pc_string = "[]"

            credal_row = credal_sets[node][pc_string]
            p_true = sample_from_interval(credal_row["True"])
            p_false = 1.0 - p_true

            # Sample the world state
            sampled_states[node] = random.random() < p_true

            cpd = TabularCPD(
                variable=node,
                variable_card=2,
                values=[[p_true], [p_false]]
            )

            model.add_cpds(cpd)
            continue

        # Case 2: Node HAS parents
        n_parents = len(parent_list)
        n_combinations = 2 ** n_parents

        # Initialize CPT rows: True row + False row
        full_values = [
            [0.0] * n_combinations,  # True probabilities
            [0.0] * n_combinations   # False probabilities
        ]

        # Iterate over all parent configurations in pgmpy order
        for idx, config in enumerate(itertools.product([False, True], repeat=n_parents)):

            pc_cfg = dict(zip(parent_list, config))
            pc_string = "[" + ", ".join(f"{k}={v}" for k, v in pc_cfg.items()) + "]"

            # Apply constraints for THIS configuration
            forced_value = apply_logical_constraints(node, pc_cfg, constraints)

            if forced_value is not None:
                # Deterministic row
                p_true = 1.0 if forced_value else 0.0
            else:
                # Sample p(True) from credal intervals for THIS configuration
                credal_row = credal_sets[node][pc_string]
                p_true = sample_from_interval(credal_row["True"])

            p_false = 1.0 - p_true

            # Write probabilities
            full_values[0][idx] = p_true
            full_values[1][idx] = p_false

        # Use the *actual* parent state that occurred in the sample
        actual_pc = {p: sampled_states[p] for p in parent_list}
        actual_config = tuple(actual_pc[p] for p in parent_list)
        actual_idx = list(itertools.product([False, True], repeat=n_parents)).index(actual_config)

        sampled_states[node] = random.random() < full_values[0][actual_idx]

        # Create TabularCPD
        parent_card = [2] * n_parents

        cpd = TabularCPD(
            variable=node,
            variable_card=2,
            values=full_values,
            evidence=parent_list,
            evidence_card=parent_card
        )

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


def ancestral_sample_bn(model: DiscreteBayesianNetwork, n_samples: int = 1):
    """
    Running ancestral (forward) sampling on a bayesian network.

    Args:
        model: DiscreteBayesianNetwork object (sampled from LCN)
        n_samples: number of samples to draw
    """

    sampler = BayesianModelSampling(model)
    
    # Draw samples from the BN
    df_samples = sampler.forward_sample(size=n_samples)
    
    # Convert each row to a dict
    #samples_list = df_samples.to_dict(orient='records')

    return df_samples
    
    #return samples_list