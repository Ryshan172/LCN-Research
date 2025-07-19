# Sampling strategies

import random
import networkx as nx


def random_from_interval(interval):
    return random.uniform(interval[0], interval[1])

def satisfies_constraints(sample, constraints):
    for constraint in constraints:
        condition = constraint["if"]
        conclusion = constraint["then"]
        if all(sample.get(k) == v for k, v in condition.items()):
            if not all(sample.get(k) == v for k, v in conclusion.items()):
                return False
    return True

def sample_node(name, parents, credal_sets, sample):
    parent_values = ', '.join([f"{p}={sample[p]}" for p in parents])
    credal_key = f"[{parent_values}]" if parents else "[]"
    
    if name not in credal_sets:
        # Default to uniform if no credal set
        return random.choice([True, False])
    
    # Try to match exact credal set first
    cset = credal_sets[name].get(credal_key)
    
    if not cset:
        # If not exact match, fallback to any available or uniform
        cset = list(credal_sets[name].values())[0]

    true_prob = random_from_interval(cset["True"])
    return random.random() < true_prob

def sample_from_lcn(lcn):
    nodes = lcn["nodes"]
    edges = lcn["edges"]
    credal_sets = lcn.get("credal_sets", {})
    constraints = lcn.get("logical_constraints", [])

    # Build DAG and get topological order
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    order = list(nx.topological_sort(G))

    while True:
        sample = {}
        for node in order:
            parents = list(G.predecessors(node))
            sample[node] = sample_node(node, parents, credal_sets, sample)

        if satisfies_constraints(sample, constraints):
            return sample  # return if valid

# Example: generate 100 samples
def generate_dataset(lcn, n=100):
    return [sample_from_lcn(lcn) for _ in range(n)]
