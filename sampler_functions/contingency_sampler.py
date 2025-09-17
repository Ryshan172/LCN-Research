import random
import itertools
import pandas as pd

# ---------- Helper functions ----------

def sample_from_interval(prob_interval):
    """
    Pick a random probability inside an interval [low, high].
    """
    low, high = prob_interval
    return random.uniform(low, high)

def choose_state(prob_true):
    """
    Bernoulli draw given probability of True.
    """
    return random.random() < prob_true

def satisfies_constraints(sample, constraints):
    """
    Check if a sampled assignment satisfies all logical constraints.
    Each constraint is of form {"if": {...}, "then": {...}}.
    """
    for c in constraints:
        cond = all(sample.get(var) == val for var, val in c["if"].items())
        if cond:
            if not all(sample.get(var) == val for var, val in c["then"].items()):
                return False
    return True


# ---------- Main sampling procedure ----------

def sample_dataset(structure, num_samples=1000):
    """
    Sample synthetic data from the given LCN.
    structure: dict with nodes, edges, credal_sets, and logical_constraints
    num_samples: number of forward samples to draw
    Returns: list of valid samples (dicts of variable assignments)
    """
    nodes = structure["nodes"]
    edges = structure["edges"]
    credal_sets = structure["credal_sets"]
    constraints = structure["logical_constraints"]

    samples = []

    for _ in range(num_samples):
        sample = {}

        # Go in topological order (here: assume given nodes list is topological enough)
        for node in nodes:
            # Find parent values
            parents = [u for u, v in edges if v == node]
            if parents:
                key = "[" + ",".join(f"{p}={sample[p]}" for p in parents) + "]"
            else:
                key = "[]"

            # Credal set for this node under parent assignment
            credal = credal_sets[node][key]

            # Pick a random prob from interval for True
            prob_true = sample_from_interval(credal["True"])
            state = choose_state(prob_true)
            sample[node] = state

        # Enforce logical constraints
        if satisfies_constraints(sample, constraints):
            samples.append(sample)

    return samples


# ---------- Aggregate into interval contingency table ----------

def aggregate_intervals(samples, structure):
    """
    Aggregate sampled data into contingency table with interval counts.
    For each (node, parent config), compute lower/upper counts per state.
    """
    nodes = structure["nodes"]
    edges = structure["edges"]

    rows = []

    for node in nodes:
        parents = [u for u, v in edges if v == node]
        # Enumerate all parent configurations
        parent_configs = list(itertools.product([False, True], repeat=len(parents)))

        for config in parent_configs:
            config_dict = dict(zip(parents, config))

            # Filter samples that match this parent config
            filtered = [s for s in samples if all(s[p] == val for p, val in config_dict.items())]

            N_total = len(filtered)

            if N_total == 0:
                continue

            # Count how many True/False observed
            count_true = sum(1 for s in filtered if s[node])
            count_false = N_total - count_true

            # Build simple intervals: here just [count, count]
            # (later you can widen to model epistemic uncertainty)
            row = {
                "node": node,
                "parent_config": ",".join(f"{p}={v}" for p, v in config_dict.items()) or "[]",
                "N_total": N_total,
                "count_false_lower": count_false,
                "count_false_upper": count_false,
                "count_true_lower": count_true,
                "count_true_upper": count_true
            }
            rows.append(row)

    return pd.DataFrame(rows)


def credal_aggregate_intervals(samples, structure):
    """
    Aggregate sampled data into contingency table with interval counts.
    For each (node, parent config), compute lower/upper counts per state.

    CHANGES from aggregate_intervals:
    - Instead of collapsing counts to a single value [count, count],
      we now use the credal set probability intervals to compute
      lower and upper bounds for counts.
    - This way the table reflects epistemic uncertainty directly
      from the model, not just the realized sample.
    """
    nodes = structure["nodes"]
    edges = structure["edges"]

    rows = []

    for node in nodes:
        parents = [u for u, v in edges if v == node]
        # Enumerate all parent configurations
        parent_configs = list(itertools.product([False, True], repeat=len(parents)))

        for config in parent_configs:
            config_dict = dict(zip(parents, config))

            # Filter samples that match this parent config
            filtered = [s for s in samples if all(s[p] == val for p, val in config_dict.items())]

            N_total = len(filtered)

            if N_total == 0:
                continue

            # ---- CHANGE: use credal set intervals for probability bounds ----
            if parents:
                key = "[" + ",".join(f"{p}={config_dict[p]}" for p in parents) + "]"
            else:
                key = "[]"

            credal = structure["credal_sets"][node][key]

            p_true_low, p_true_high = credal["True"]
            p_false_low, p_false_high = credal["False"]

            # Convert probability intervals into count intervals
            count_true_lower = int(N_total * p_true_low)
            count_true_upper = int(N_total * p_true_high)
            count_false_lower = int(N_total * p_false_low)
            count_false_upper = int(N_total * p_false_high)

            # ---- Result row with widened intervals ----
            row = {
                "node": node,
                "parent_config": ",".join(f"{p}={v}" for p, v in config_dict.items()) or "[]",
                "N_total": N_total,
                "count_false_lower": count_false_lower,
                "count_false_upper": count_false_upper,
                "count_true_lower": count_true_lower,
                "count_true_upper": count_true_upper
            }
            rows.append(row)

    return pd.DataFrame(rows)



def run_aggregate_sampler(lcn):
    samples = sample_dataset(lcn, num_samples=1000)
    print(samples)
    # df = aggregate_intervals(samples, lcn)
    df = credal_aggregate_intervals(samples, lcn)

    return df


# ---------- Example usage ----------

if __name__ == "__main__":
    # Example LCN structure
    lcn = {
        "nodes": ["X1", "X2", "X3", "X4", "X5"],
        "edges": [["X1","X2"],["X1","X4"],["X1","X5"],["X2","X3"]],
        "credal_sets": {
            "X1": {"[]": {"True":[0.56,0.86],"False":[0.14,0.44]}},
            "X2": {"[X1=True]":{"True":[1.0,1.0],"False":[0.0,0.0]},
                   "[X1=False]":{"True":[0.14,0.44],"False":[0.56,0.86]}},
            "X3": {"[X2=True]":{"True":[0.41,0.71],"False":[0.29,0.59]},
                   "[X2=False]":{"True":[0.14,0.44],"False":[0.56,0.86]}},
            "X4": {"[X1=True]":{"True":[0.1,0.4],"False":[0.6,0.9]},
                   "[X1=False]":{"True":[0.0,0.0],"False":[1.0,1.0]}},
            "X5": {"[X1=True]":{"True":[0.14,0.44],"False":[0.56,0.86]},
                   "[X1=False]":{"True":[0.24,0.54],"False":[0.46,0.76]}}
        },
        "logical_constraints": [
            {"if":{"X1":True}, "then":{"X2":True}},
            {"if":{"X1":False}, "then":{"X4":False}}
        ]
    }

    samples = sample_dataset(lcn, num_samples=1000)
    df = aggregate_intervals(samples, lcn)

    print(df.head())
    df.to_csv("lcn_dataset.csv", index=False)
