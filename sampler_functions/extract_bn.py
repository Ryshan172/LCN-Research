import random
import itertools
import pandas as pd

def sample_bn_from_lcn(lcn):
    """
    Given a Logical Credal Network (LCN), return a single sampled Bayesian Network (BN)
    with concrete Conditional Probability Tables (CPTs).
    
    Steps:
    - For each node, determine its parents and all possible parent value combinations.
    - For each parent configuration, sample a probability for the node being True
      from its credal interval.
    - Apply logical constraints to override probabilities if the configuration triggers them.
    - Store the resulting CPT for the node.
    """
    # Initialize BN structure with same nodes and edges as LCN
    bn = {"nodes": lcn["nodes"], "edges": lcn["edges"], "cpts": {}}
    credal_sets = lcn["credal_sets"]
    constraints = lcn.get("logical_constraints", [])  # optional logical rules

    for node in bn["nodes"]:
        node_cpt = {}  # store CPT for this node
        # Determine parent nodes
        parents = sorted([u for u, v in bn["edges"] if v == node])
        # Enumerate all possible parent value combinations (True/False)
        parent_configs = [
            dict(zip(parents, cfg)) for cfg in itertools.product([False, True], repeat=len(parents))
        ]

        for config in parent_configs:
            # Create key string matching LCN credal set format, e.g., "[X1=True, X2=False]"
            key = "[" + ", ".join(f"{p}={config[p]}" for p in parents) + "]" if parents else "[]"
            prob_true_interval = credal_sets[node][key]["True"]
            
            # Sample a concrete probability for True from the credal interval
            prob_true = random.uniform(*prob_true_interval)

            # Check logical constraints and override probability if applicable
            for c in constraints:
                # If this parent configuration satisfies the "if" part of the constraint
                if all(config.get(var, None) == val for var, val in c["if"].items()):
                    # Apply the "then" part of the constraint for this node
                    for var, val in c["then"].items():
                        if var == node:
                            prob_true = 1.0 if val else 0.0  # force True/False

            # Store the CPT entry for this parent configuration
            node_cpt[key] = {"True": prob_true, "False": 1 - prob_true}

        # Save the CPT for this node in the BN
        bn["cpts"][node] = node_cpt

    return bn


def sample_multiple_bns(lcn, n=10):
    """
    Sample multiple Bayesian Networks from a single LCN.
    Each BN has CPTs sampled independently from the LCN's credal intervals.
    
    Args:
    - lcn: Logical Credal Network dictionary
    - n: number of BNs to sample
    
    Returns:
    - List of n Bayesian Networks (dicts with nodes, edges, CPTs)
    """
    return [sample_bn_from_lcn(lcn) for _ in range(n)]
