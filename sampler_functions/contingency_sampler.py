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
            # Find parent values (sorted to fix multiple incoming edge issue)
            parents = sorted([u for u, v in edges if v == node])

            if parents:
                key = "[" + ", ".join(f"{p}={sample[p]}" for p in parents) + "]" 
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
                "parent_config": ", ".join(f"{p}={v}" for p, v in config_dict.items()) or "[]",
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
                key = "[" + ", ".join(f"{p}={config_dict[p]}" for p in parents) + "]"
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
                "parent_config": ", ".join(f"{p}={v}" for p, v in config_dict.items()) or "[]",
                "N_total": N_total,
                "count_false_lower": count_false_lower,
                "count_false_upper": count_false_upper,
                "count_true_lower": count_true_lower,
                "count_true_upper": count_true_upper
            }
            rows.append(row)

    return pd.DataFrame(rows)



def run_aggregate_sampler(lcn):
    forward_samples = sample_dataset(lcn, num_samples=1000)

    # Convert to dataframe
    samples_df = pd.DataFrame(forward_samples)

    # Save samples
    samples_save_file = "lcn_forward_samples.csv"

    samples_df.to_csv(f"datasets/contingency_samples/{samples_save_file}", index=False)

    # Compute aggregate contingency table
    aggregate_df = credal_aggregate_intervals(forward_samples, lcn)

    return aggregate_df, samples_df