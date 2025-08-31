import random

def sample_from_interval(interval):
    low, high = interval
    return random.uniform(low, high)

def forward_sample(lcn):
    sample = {}

    # topological order (can compute dynamically, but here it's fixed)
    order = ["A", "B", "C", "E", "D"]

    for node in order:
        parents = [src for src, dst in lcn["edges"] if dst == node]
        parent_values = {p: sample[p] for p in parents}

        # Find credal set entry
        credal_table = lcn["credal_sets"][node]
        key = str([f"{p}={parent_values[p]}" for p in parents]) if parents else "[]"

        probs = {}
        for val in ["True", "False"]:
            interval = credal_table[key][val]
            probs[val] = sample_from_interval(interval)

        # Normalize (intervals may not sum to 1 after random pick)
        total = probs["True"] + probs["False"]
        probs["True"] /= total
        probs["False"] /= total

        # Draw sample
        sample[node] = random.choices(["True", "False"], weights=[probs["True"], probs["False"]])[0]

    # Apply logical constraints
    for constr in lcn["logical_constraints"]:
        if_node, if_val = list(constr["if"].items())[0]
        then_node, then_val = list(constr["then"].items())[0]
        if sample[if_node] == str(if_val):
            sample[then_node] = str(then_val)

    return sample
