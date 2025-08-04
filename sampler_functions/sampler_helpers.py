import random
import networkx as nx

def value_from_interval(interval, mode="random"):
    """
    Sample a value from an interval using the given mode.
    """
    if mode == "lower":
        return interval[0]
    elif mode == "upper":
        return interval[1]
    elif mode == "midpoint":
        return (interval[0] + interval[1]) / 2
    else:  # default: random
        return random.uniform(interval[0], interval[1])


def sample_node(name, parents, credal_sets, sample, mode="random"):
    """
    Sample a Boolean value for a node based on its credal set and current sample state.
    """
    if name not in credal_sets:
        return random.choice([True, False])  # fallback: uniform

    # Sort parent names to ensure key consistency
    if parents:
        parent_values = ', '.join([f"{p}={sample[p]}" for p in sorted(parents)])
        credal_key = f"[{parent_values}]"
    else:
        credal_key = "[]"

    node_credal = credal_sets[name]
    cset = node_credal.get(credal_key)

    if not cset:
        raise ValueError(f"No credal set found for node '{name}' with key {credal_key}")

    true_prob = value_from_interval(cset["True"], mode)
    return random.random() < true_prob


def satisfies_constraints(sample, constraints):
    """
    Check if the sample satisfies all logical constraints.
    """
    for constraint in constraints:
        condition = constraint["if"]
        conclusion = constraint["then"]
        if all(sample.get(k) == v for k, v in condition.items()):
            if not all(sample.get(k) == v for k, v in conclusion.items()):
                return False
    return True


def sample_from_lcn(lcn, mode="random"):
    """
    Sample one valid assignment from the Logical Credal Network.
    """
    nodes = lcn["nodes"]
    edges = lcn["edges"]
    credal_sets = lcn.get("credal_sets", {})
    constraints = lcn.get("logical_constraints", [])

    # Build graph and determine topological order
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    order = list(nx.topological_sort(G))

    # Rejection sampling until constraints are satisfied
    while True:
        sample = {}
        for node in order:
            parents = list(G.predecessors(node))
            sample[node] = sample_node(node, parents, credal_sets, sample, mode)

        if satisfies_constraints(sample, constraints):
            return sample


def generate_interval_dataset(lcn, n=100, mode="random"):
    """
    Generate a dataset of valid samples from the LCN.
    Default mode is random
    """
    return [sample_from_lcn(lcn, mode) for _ in range(n)]
