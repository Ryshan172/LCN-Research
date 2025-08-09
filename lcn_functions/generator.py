import random
import string
import itertools
import json
from collections import deque

def generate_lcn(size, interval_width=0.2, num_constraints=None, constraint_chaining=False):
    """
    Generate a Logical Credal Network (LCN) with:
    - A specified number of nodes (size)
    - Random DAG structure
    - Interval-valued probability parameters
    - Logical constraints consistent with the DAG
    - Optional chaining: constraints propagate through DAG

    Args:
        size (int): Number of nodes in the LCN.
        interval_width (float): Width of probability intervals (0 < width <= 1).
        num_constraints (int or None): Number of logical constraints. If None, chosen randomly.
        constraint_chaining (bool): If True, constraints are transitive through the DAG.

    Returns:
        dict: LCN structure.
    """
    
    # === Generate node names ===
    nodes = list(string.ascii_uppercase[:size])
    
    # === Generate a random DAG ===
    edges = []
    for i in range(size):
        for j in range(i + 1, size):
            if random.random() < 0.4:
                edges.append([nodes[i], nodes[j]])
    
    possible_pairs = [(p, c) for p, c in edges]
    if not possible_pairs:
        edges.append([nodes[0], nodes[1]])
        possible_pairs.append((nodes[0], nodes[1]))
    
    # === Generate constraints consistent with DAG ===
    if num_constraints is None:
        num_constraints = random.randint(1, max(1, size // 2))
    
    logical_constraints = []
    for _ in range(num_constraints):
        parent, child = random.choice(possible_pairs)
        val_src = random.choice([True, False])
        val_tgt = random.choice([True, False])
        logical_constraints.append({
            "if": {parent: val_src},
            "then": {child: val_tgt}
        })
    
    # Optionally expand constraints via chaining
    if constraint_chaining:
        logical_constraints = propagate_constraints(logical_constraints, edges)
    
    # === Generate credal sets ===
    credal_sets = {}
    for node in nodes:
        parents = [p for p, c in edges if c == node]
        credal_sets[node] = {}
        
        if not parents:
            cond_key = "[]"
            credal_sets[node][cond_key] = create_conditional_probs(
                interval_width, node, {}, logical_constraints
            )
        else:
            parent_combos = list(itertools.product([True, False], repeat=len(parents)))
            for combo in parent_combos:
                condition_dict = {p: v for p, v in zip(parents, combo)}
                cond_key = "[" + ", ".join(f"{p}={str(v)}" for p, v in condition_dict.items()) + "]"
                credal_sets[node][cond_key] = create_conditional_probs(
                    interval_width, node, condition_dict, logical_constraints
                )
    
    return {
        "nodes": nodes,
        "edges": edges,
        "credal_sets": credal_sets,
        "logical_constraints": logical_constraints
    }

def propagate_constraints(constraints, edges):
    """
    Expand constraints through DAG connections.
    Example:
      If A=True => B=True and B=True => C=False,
      then A=True => C=False is added.
    """
    new_constraints = constraints.copy()
    added = True
    while added:
        added = False
        for c1 in list(new_constraints):
            for c2 in list(new_constraints):
                then_node, then_val = list(c1["then"].items())[0]
                if_node, if_val = list(c2["if"].items())[0]
                if then_node == if_node and then_val == if_val:
                    new_constraint = {"if": c1["if"], "then": c2["then"]}
                    if new_constraint not in new_constraints:
                        new_constraints.append(new_constraint)
                        added = True
    return new_constraints

def create_conditional_probs(interval_width, node, condition_dict, constraints):
    """
    Create probability intervals respecting logical constraints.
    """
    for cons in constraints:
        if_node, if_val = list(cons["if"].items())[0]
        then_node, then_val = list(cons["then"].items())[0]
        
        if node == then_node and if_node in condition_dict:
            if condition_dict[if_node] == if_val:
                if then_val is True:
                    return {"True": [1.0, 1.0], "False": [0.0, 0.0]}
                else:
                    return {"True": [0.0, 0.0], "False": [1.0, 1.0]}
    
    return {
        "True": random_interval(interval_width),
        "False": random_interval(interval_width)
    }

def random_interval(width):
    low = round(random.uniform(0, 1 - width), 2)
    high = round(low + width, 2)
    return [low, high]

# ==== Example ====
if __name__ == "__main__":
    lcn = generate_lcn(size=5, interval_width=0.3, num_constraints=2, constraint_chaining=True)
    print(json.dumps(lcn, indent=2))
