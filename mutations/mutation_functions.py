import copy
import random
import networkx as nx
from pgmpy.base import DAG


"""
Functionality for performing constraint-aware mutations during structure learning

Contains functions for different contraint-aware mutations that work differently


Theoretical Assumptions
-----------------------

Assumption 1:
Logical feasibility is approximated structurally.

The theoretical definition of feasibility is

    (M, Θ) ⊨ C

where M is the graph structure, Θ are the credal parameters,
and C is the set of logical constraints.

To preserve the local-search complexity analysis,
the implementation does NOT perform full probabilistic
constraint verification after each mutation.

Instead, logical constraints are interpreted as
dependency-preservation requirements.

Example:

    X2=True -> X3=False

is interpreted as

    X2 should remain capable of influencing X3.

This provides a computationally inexpensive approximation
to logical feasibility.


Assumption 2:
Constraint evaluation is local.

Only constraints involving variables directly affected
by a mutation are evaluated.

If a mutation modifies

    X2 -> X3

then only constraints mentioning X2 or X3
are considered.

This follows the locality assumption used in the
complexity analysis.


Assumption 3:
Constraint evaluation is O(1).

The number of constraints associated with a local
mutation is assumed bounded by a small constant.

Therefore constraint verification contributes only
constant-time overhead per candidate evaluation.
"""

#-------------- Helper Functions --------------------------------------
def extract_constraint_dependencies(logical_constraints):
    """
    Converts logical constraints into dependency pairs.

    Example
    -------

    X2=True -> X3=False

    becomes

    ('X2', 'X3')

    These pairs are NOT interpreted as mandatory edges.

    They merely indicate that some dependency relationship
    should remain present after mutation.
    """

    dependencies = set()

    for rule in logical_constraints:
        for source in rule["if"].keys():
            for target in rule["then"].keys():
                dependencies.add((source, target))

    return dependencies


def affected_constraints(
    logical_constraints,
    affected_nodes
):
    """
    Returns only constraints involving variables
    touched by the mutation.

    This implements local constraint evaluation
    assumed in the theoretical analysis.
    """

    relevant = []

    for rule in logical_constraints:

        variables = (
            set(rule["if"].keys())
            |
            set(rule["then"].keys())
        )

        if variables.intersection(affected_nodes):
            relevant.append(rule)

    return relevant


def local_constraint_violation(candidate_state, affected_nodes, constraint_index):
    """
    Returns True if ANY required directed dependency edge is missing.

    This matches your theory:

        (M, Θ) ⊨ C

    interpreted structurally as:

        required directed edges must exist in M
    """

    logical_constraints = candidate_state["logical_constraints"]

    relevant_constraints = get_relevant_constraints(
        constraint_index,
        logical_constraints,
        affected_nodes
    )

    required_edges = extract_constraint_dependencies(relevant_constraints)

    edge_set = set(candidate_state["edges"])

    for source, target in required_edges:
        if (source, target) not in edge_set:
            return True

    return False


def get_relevant_constraints(constraint_index, logical_constraints, affected_nodes):
    """
    Returns only constraints touching affected nodes.

    Complexity assumption:
        O(1)-amortised via index lookup
    """

    relevant_ids = set()

    for v in affected_nodes:
        if v in constraint_index:
            relevant_ids |= constraint_index[v]

    return [logical_constraints[i] for i in relevant_ids]



#--------------- Mutation Functions------------------------------------
def standard_mutation(lcn_state, mutation_type="edge_add", max_attempts=10):

    nodes = lcn_state["nodes"]
    edges = copy.deepcopy(lcn_state["edges"])

    for _ in range(max_attempts):

        candidate_edges = copy.deepcopy(edges)

        if mutation_type == "edge_add":
            a, b = random.sample(nodes, 2)

            if (a, b) not in candidate_edges:
                candidate_edges.append((a, b))

            affected_nodes = {a, b}

        elif mutation_type == "edge_delete" and candidate_edges:
            a, b = random.choice(candidate_edges)
            candidate_edges.remove((a, b))
            affected_nodes = {a, b}

        elif mutation_type == "edge_flip" and candidate_edges:
            i = random.randint(0, len(candidate_edges) - 1)
            a, b = candidate_edges[i]
            candidate_edges[i] = (b, a)
            affected_nodes = {a, b}

        else:
            return lcn_state

        candidate = copy.deepcopy(lcn_state)
        candidate["edges"] = candidate_edges

        return candidate

    return lcn_state



def constraint_aware_mutation(
    lcn_state,
    constraint_index,
    mutation_type="edge_add",
    max_attempts=10
):
    """
    Instead of allowing every add/remove/reverse operation, 
    mutations could first check whether the resulting structure violates 
    logical constraints.

    Examples:
    - Preventing edge removals that disconnect logically dependent nodes 
    - Prevent reversals that contradict implied logic direction
    - Enforce required edges to not break constraints

    Functionality: 
        Pre-check constraint-aware mutation.

        - Generates candidate mutations
        - Rejects those violating local constraints
        - Ensures at least one valid mutation is returned
    """

    nodes = lcn_state["nodes"]
    edges = copy.deepcopy(lcn_state["edges"])

    for _ in range(max_attempts):

        candidate_edges = copy.deepcopy(edges)

        if mutation_type == "edge_add":
            a, b = random.sample(nodes, 2)
            candidate_edges.append((a, b))
            affected_nodes = {a, b}

        elif mutation_type == "edge_delete" and candidate_edges:
            a, b = random.choice(candidate_edges)
            candidate_edges.remove((a, b))
            affected_nodes = {a, b}

        elif mutation_type == "edge_flip" and candidate_edges:
            i = random.randint(0, len(candidate_edges) - 1)
            a, b = candidate_edges[i]
            candidate_edges[i] = (b, a)
            affected_nodes = {a, b}

        else:
            continue

        candidate = copy.deepcopy(lcn_state)
        candidate["edges"] = candidate_edges

        if not local_constraint_violation(candidate, affected_nodes, constraint_index):
            return candidate, {"affected_nodes": affected_nodes}

    return lcn_state, {"affected_nodes": set()}


def post_mutation_contraint_repair(
    lcn_state,
    constraint_index,
    mutation_type="edge_add",
    max_attempts=10,
    max_repairs=3
):
    """
    POST-MUTATION REPAIR (STRICT LOCALITY VERSION)

    Pipeline:
    1. Apply mutation (standard or constraint-aware)
    2. Identify affected nodes (local footprint only)
    3. Query constraints ONLY via precomputed index
    4. Repair only missing dependency edges
    """

    candidate, meta = constraint_aware_mutation(
        lcn_state,
        constraint_index,
        mutation_type,
        max_attempts
    )

    affected_nodes = meta["affected_nodes"]

    # No affected region → no repair needed
    if not affected_nodes:
        return candidate

    edge_set = set(candidate["edges"])


    # STEP 2: local repair loop
    for _ in range(max_repairs):

        # TRUE LOCAL CHECK (index-based only)
        if not local_constraint_violation(candidate, affected_nodes, constraint_index):
            return candidate

        # IMPORTANT:
        # only index-based retrieval (no full scan ever)
        relevant_constraints = get_relevant_constraints(
            constraint_index,
            candidate["logical_constraints"],
            affected_nodes
        )

        required_edges = extract_constraint_dependencies(relevant_constraints)

        for source, target in required_edges:

            if (source, target) not in edge_set:
                candidate["edges"].append((source, target))
                edge_set.add((source, target))

    return candidate