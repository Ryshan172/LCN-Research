import random
import string
import itertools
import json
from collections import deque

def generate_lcn(size, interval_width=0.2, num_constraints=None, constraint_chaining=False, edge_prob=0.4):
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
        edge_prob: chance of an edge being added e.g. 0.4 = 40 % chance, 0.1 is sparse, 0.9 is denser

    Returns:
        dict: LCN structure.
    """
    
    # Generate node names (e.g A, B, C) depending on size param
    nodes = list(string.ascii_uppercase[:size])
    
    """
    Generate a random DAG. 
    - Goes through every possible edge given the size range
    - uses edge_prob weighting to decide if that edge should exist and adds it
    - E.g. Run 1 could be A -> B, C - D, or A -> C depending on prob
    - If the random number falls below `edge_prob`, the edge is added. Otherwise it is skipped
    - i < j ensures acyclicity 
    """
    edges = []
    for i in range(size):
        for j in range(i + 1, size):
            if random.random() < edge_prob:
                edges.append([nodes[i], nodes[j]])
    
    possible_pairs = [(p, c) for p, c in edges]
    if not possible_pairs:
        edges.append([nodes[0], nodes[1]])
        possible_pairs.append((nodes[0], nodes[1]))
    

    # Generate LCN constraints consistent with DAG (either defined or random)
    if num_constraints is None:
        num_constraints = random.randint(1, max(1, size // 2))
    
    logical_constraints = []
    for constraint_index in range(num_constraints):
        # Randomly pick an existing edge (parent → child) from the DAG.
        parent, child = random.choice(possible_pairs)
        # Randomly decide whether the parent condition is True or False.
        val_parent = random.choice([True, False])
        # Randomly decide whether the child should be True or False given the parent condition.
        val_child = random.choice([True, False])
        # E.g. If A is True, then B must be False
        logical_constraints.append({
            "if": {parent: val_parent},
            "then": {child: val_child}
        })
    
    # Optionally expand constraints via chaining throughout graph
    if constraint_chaining:
        logical_constraints = propagate_constraints(logical_constraints, edges)
    

    """
    Credal Set generation.
    - Loops over each node in the DAG.
    - If a node has no parents, it gets one unconditional credal set (`[]`).
    - If it has parents, it creates one credal set per combination of parent truth values.
        - Example: If node `C` has parents `[A, B]`, it creates credal sets for  
            `[A=True, B=True]`, `[A=True, B=False]`, `[A=False, B=True]`, `[A=False, B=False]`.
    - Each case calls `create_conditional_probs(...)` to actually fill in the interval-valued probabilities, while respecting logical constraints.
    """
    # Dictionary to hold credal sets for all nodes
    credal_sets = {}

    for node in nodes:
        # Find all parents of the current node
        parents = [p for p, c in edges if c == node]
        # Initialize credal set for node
        credal_sets[node] = {}
        
        if not parents:
            # If node has no parents (root node) than condition key is empty
            cond_key = "[]"
            credal_sets[node][cond_key] = create_conditional_probs(
                interval_width, node, {}, logical_constraints
            )
        else:
            # If the node has parents, generate all possible truth assignments for the parents
            parent_combos = list(itertools.product([True, False], repeat=len(parents)))
            for combo in parent_combos:
                # Map parents to their True/False values
                condition_dict = {p: v for p, v in zip(parents, combo)}
                cond_key = "[" + ", ".join(f"{p}={str(v)}" for p, v in condition_dict.items()) + "]"

                # Create conditional credal set for this configuration
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

    Returns expanded set of constraints        
    """
    # Start with existing constraints
    new_constraints = constraints.copy()
    # Flag for tracking adding
    added = True

    while added:
        # Reset flag
        added = False
        # Check every constraint as a possible starting rule
        for c1 in list(new_constraints):
            # Compare it with every other constraint
            for c2 in list(new_constraints):
                # Extract 'then' from c1
                then_node, then_val = list(c1["then"].items())[0]
                # Extract "if" part of c2
                if_node, if_val = list(c2["if"].items())[0]

                # If the "then" of c1 matches the "if" of c2, they can be chained
                if then_node == if_node and then_val == if_val:
                    # Build chained rule
                    new_constraint = {"if": c1["if"], "then": c2["then"]}

                    # Add new rule only if it's not already in the list
                    if new_constraint not in new_constraints:
                        new_constraints.append(new_constraint)
                        # End round
                        added = True
         
    return new_constraints


def create_conditional_probs(interval_width, node, condition_dict, constraints):
    """
    Create probability intervals for a node given its parent values,
    while respecting logical constraints if they apply.
    """
    for constr in constraints:
        # Extract 'if' variable and value
        if_node, if_val = list(constr["if"].items())[0]
        # Extract 'then' variable and value
        then_node, then_val = list(constr["then"].items())[0]

        # If this node is constrained AND the condition matches parent assignment
        if node == then_node and if_node in condition_dict:
            # Enforce the constraint strictly: probability is fixed (no interval)
            if condition_dict[if_node] == if_val:
                if then_val is True:
                    # Node must be true
                    return {"True": [1.0, 1.0], "False": [0.0, 0.0]}
                else:
                    # Node must be false
                    return {"True": [0.0, 0.0], "False": [1.0, 1.0]}
    
    # If no constraint applies, assign a random interval for both True/False outcomes
    return {
        "True": random_interval(interval_width),
        "False": random_interval(interval_width)
    }


def random_interval(width):
    # Pick a random lower bound between 0 and (1 - width)
    low = round(random.uniform(0, 1 - width), 2)
    # Upper bound is fixed distance 'width' above the lower bound
    high = round(low + width, 2)
    return [low, high]

# # ==== Example ====
# if __name__ == "__main__":
#     # lcn = generate_lcn(size=5, interval_width=0.3, num_constraints=2, constraint_chaining=True)
#     # print(json.dumps(lcn, indent=2))
