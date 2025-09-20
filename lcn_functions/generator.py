import random
import string
import itertools
import json
from collections import deque
from .lcn_check import check_all_nodes_connected
import numpy as np


def generate_lcn(size, interval_width=0.2, num_constraints=None, constraint_chaining=False, edge_prob=0.4, in_degree=1, dist_type="beta"):
    """
    Generate a Logical Credal Network (LCN) with:
    - A specified number of nodes (size)
    - Random DAG structure
    - Interval-valued probability parameters
    - Logical constraints consistent with the DAG
    - Optional chaining: constraints propagate through DAG

    Args:
        - size (int): Number of nodes in the LCN.
        - interval_width (float): Width of probability intervals (0 < width <= 1).
        - num_constraints (int or None): Number of logical constraints. If None, chosen randomly.
        - constraint_chaining (bool): If True, constraints are transitive through the DAG.
        - edge_prob: chance of an edge being added e.g. 0.4 = 40 % chance, 0.1 is sparse, 0.9 is denser
        - in_degree: Integer that determines that number of incoming edges a node will have. e.g 1 means only one edge

    Note: in_degree directly determines how many parents a node can have.
    
    Returns:
        dict: LCN structure.
    """
    
    # Generate node names (e.g A, B, C) depending on size param
    #nodes = list(string.ascii_uppercase[:size])

    # Using X1, X2 ...Xn node names instead
    nodes = [f"X{i+1}" for i in range(size)]
    
    """
    Generate a random DAG. 
    - Goes through every possible edge given the size range
    - uses edge_prob weighting to decide if that edge should exist and adds it
    - E.g. Run 1 could be A -> B, C - D, or A -> C depending on prob
    - If the random number falls below `edge_prob`, the edge is added. Otherwise it is skipped
    - i < j ensures acyclicity 
    - Runs in a while loop to ensure no edges are unconnected
    - Note: Remove While loop and check if unconnected edges are allowed
    """

    # TODO: Implement in_degree parameter

    while True:
        edges = []
        # For managing number of incoming edges to node
        in_deg_count = {n: 0 for n in nodes}

        for i in range(size):
            for j in range(i + 1, size):
                # Only add edge if child (nodes[j]) has fewer than allowed parents
                if random.random() < edge_prob and in_deg_count[nodes[j]] < in_degree:
                    edges.append([nodes[i], nodes[j]])
                    in_deg_count[nodes[j]] += 1
        
        # fallback in case none generated
        if not edges:
            if in_deg_count[nodes[1]] < in_degree:
                edges.append([nodes[0], nodes[1]])
                in_deg_count[nodes[1]] += 1


        # check connectivity
        if check_all_nodes_connected(nodes, edges):
            break   # stop regenerating once connected
    

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
            # Note: Changed to us randomised interval withs function
            credal_sets[node][cond_key] = create_random_width_conditional_probs(
                interval_width, node, {}, logical_constraints, dist_type
            )
        else:
            # If the node has parents, generate all possible truth assignments for the parents
            parent_combos = list(itertools.product([True, False], repeat=len(parents)))
            for combo in parent_combos:
                # Map parents to their True/False values
                condition_dict = {p: v for p, v in zip(parents, combo)}
                cond_key = "[" + ", ".join(f"{p}={str(v)}" for p, v in condition_dict.items()) + "]"

                # Create conditional credal set for this configuration
                credal_sets[node][cond_key] = create_random_width_conditional_probs(
                    interval_width, node, condition_dict, logical_constraints, dist_type
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


def create_conditional_probs(interval_width, node, condition_dict, constraints, allow_overlap=False):
    """
    Create probability intervals for a node given its parent values,
    while respecting logical constraints if they apply.

    Args:
        interval_width (float): Width of probability interval.
        node (str): Node name.
        condition_dict (dict): Parent assignments for this node.
        constraints (list): Logical constraints.
        allow_overlap (bool): If True, True/False intervals may overlap.
                              If False, they are complementary (non-overlapping).
    """

    # Check all constraints for full parent assignment match 
    for constr in constraints:
        then_node, then_val = list(constr["then"].items())[0]

        # Only consider constraints that apply to this node
        if node != then_node:
            continue

        # Use all() to check if all 'if' conditions match the parent assignment
        if all(condition_dict.get(k) == v for k, v in constr["if"].items()):
            # Apply the constraint strictly: probability is fixed (no interval)
            if then_val is True:
                return {"True": [1.0, 1.0], "False": [0.0, 0.0]}
            else:
                return {"True": [0.0, 0.0], "False": [1.0, 1.0]}

    # Generate random interval for True
    true_low, true_high = random_interval(interval_width)

    if allow_overlap:
        # ALLOW OVERLAP: Random interval for False independently
        false_low, false_high = random_interval(interval_width)
    else:
        # NO OVERLAP: False interval is complementary to True
        false_low = round(1 - true_high, 2)
        false_high = round(1 - true_low, 2)

    return {
        "True": [true_low, true_high],
        "False": [false_low, false_high]
    }


def create_random_width_conditional_probs(interval_width, node, condition_dict, constraints, dist_type="beta", beta_params=(2,2), allow_overlap=False,):
    """
    Create probability intervals for a node given its parent values,
    while respecting logical constraints if they apply.

    Args:
        interval_width (float): Default/fallback width of probability interval.
        node (str): Node name.
        condition_dict (dict): Parent assignments for this node.
        constraints (list): Logical constraints.
        allow_overlap (bool): If True, True/False intervals may overlap.
                              If False, they are complementary (non-overlapping).
        dist_type (str): Distribution type for mean probability: "beta", "gaussian", "uniform", "triangular".
        beta_params (tuple): (alpha, beta) for Beta distribution.
    
    Interval Width: 
    - Instead of fixed-width random interval, uses Beta-based mean + random width
    - Instead of always using a fixed interval width, the code now samples a mean probability 
        from a Beta distribution and a random width from a Gaussian.
    - This makes intervals more diverse and realistic, reducing bias by allowing some credal sets 
        to be narrow (high confidence) and others wide (high uncertainty).
    
    - Larger interval_width → on average wider intervals.
    - Smaller interval_width → on average narrower intervals.
    - But every individual interval is slightly different, adding diversity and avoiding bias.

    Using a Beta distribution lets you randomly generate probabilities within [0,1] while controlling the shape, 
    so some values cluster near 0, 0.5, or 1, creating realistic variability in the credal sets.
    """

    # Check all constraints for full parent assignment match 
    for constr in constraints:
        then_node, then_val = list(constr["then"].items())[0]

        # Only consider constraints that apply to this node
        if node != then_node:
            continue

        # Use all() to check if all 'if' conditions match the parent assignment
        if all(condition_dict.get(k) == v for k, v in constr["if"].items()):
            # Apply the constraint strictly: probability is fixed (no interval)
            if then_val is True:
                return {"True": [1.0, 1.0], "False": [0.0, 0.0]}
            else:
                return {"True": [0.0, 0.0], "False": [1.0, 1.0]}
            

    if dist_type == "beta":
        alpha, beta = beta_params
        p = np.random.beta(alpha, beta)
    elif dist_type == "gaussian":
        p = random.gauss(0.5, 0.15)
        p = min(max(p, 0.0), 1.0)
    elif dist_type == "uniform":
        p = random.uniform(0.0, 1.0)
    elif dist_type == "triangular":
        p = random.triangular(0.0, 1.0, 0.5)
    else:
        raise ValueError(f"Unknown dist_type: {dist_type}")


    # Sample width and compute interval
    width = max(0.01, min(0.5, random.gauss(interval_width, 0.05)))
    true_low = max(0.0, p - width / 2)
    true_high = min(1.0, p + width / 2)

    if allow_overlap:
        # Independent False interval
        false_interval = create_random_width_conditional_probs(interval_width, node, condition_dict,
                                                               [], allow_overlap=False,
                                                               dist_type=dist_type, beta_params=beta_params)
        false_low, false_high = false_interval["True"]
    else:
        # Complementary False interval
        false_low = round(1 - true_high, 2)
        false_high = round(1 - true_low, 2)

    return {
        "True": [round(true_low, 2), round(true_high, 2)],
        "False": [round(false_low, 2), round(false_high, 2)]
    }


def random_interval(width):
    # Pick a random lower bound between 0 and (1 - width)
    low = round(random.uniform(0, 1 - width), 2)
    # Upper bound is fixed distance 'width' above the lower bound
    high = round(low + width, 2)
    return [low, high]