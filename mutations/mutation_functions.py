"""Functionality for performing constraint-aware mutations during structure learning.

The three operators make single-edge additions, deletions or reversals.
Standard mutation checks the graph structure. Constraint-aware mutation also
rejects edits that remove required edges. Post-mutation repair tries to
restore the missing edges before checking the resulting graph.

Theoretical assumptions
-----------------------
1. Structural feasibility. In the notation (M, Θ) ⊨ C, M is the graph
   structure, Θ the credal parameters and C the supplied rules. The
   operators check the structural part of the problem: a candidate must be
   a DAG on the supplied nodes and respect the parent cap when one is set.
   For the constrained RQ2 strategies, each rule also supplies required
   directed edges. For example, X2=True -> X3=False requires X2 -> X3.
   Presence of that edge alone cannot enforce the stated truth values;
   that would also require restrictions on the probability parameters.

2. Local rule evaluation. From a feasible starting graph, a single edit
   can remove a required edge only at the edited endpoints. Rejection uses
   the variable-to-rule index to check rules mentioning either endpoint.
   RQ2 repair uses the same index to find missing edges, then checks the
   complete required set and the DAG conditions on the repaired graph.

3. Cost of constraint verification. Looking up one variable in the index
   takes expected constant time. Reading its list of rules and expanding
   their antecedent/consequent pairs costs time proportional to the work
   involved. Even with a bounded number of short rules per endpoint,
   copying the state, building the edge set and checking the DAG can
   depend on the size of the graph.
"""

import copy
import random

import networkx as nx


def extract_constraint_dependencies(logical_constraints):
    """Convert logical constraints into a set of required directed edges.

    For each rule, pair every variable in its "if" part with every variable
    in its "then" part. For example, X2=True -> X3=False contributes
    ('X2', 'X3'). Different rules can contribute the same pair. The set
    records the required direction once; the literal truth values do not
    enter the edge-presence check.
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
    """Return True if any required edge of a rule touching affected_nodes is missing.

    For an edit of X2 -> X3, retrieve the rules mentioning X2 or X3,
    extract their directed pairs and test membership in the candidate's
    edge set. False means every locally required edge is present. This is
    the structural test used by rejection; it assumes an initially feasible
    graph when used to preserve feasibility across single-edge edits.
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


def get_relevant_constraints(
    constraint_index,
    logical_constraints,
    affected_nodes
):
    """Return the rules mentioning at least one of the affected nodes.

    Use the precomputed index to retrieve each node's incident rules.
    A rule mentioning both endpoints appears under both keys, so return
    that rule object only once. The lookup cost depends on the number of
    entries retrieved. When no index is supplied, scan logical_constraints
    for rules touching the affected nodes.
    """
    # Changed: Fall back to scanning incident rules when the caller has no index.
    if constraint_index is None:
        return affected_constraints(logical_constraints, affected_nodes)

    relevant_constraints = []

    for v in affected_nodes:

        if v in constraint_index:
            relevant_constraints.extend(
                constraint_index[v]
            )

    # remove duplicates
    unique_constraints = []
    seen = set()

    for rule in relevant_constraints:

        rule_id = id(rule)

        if rule_id not in seen:
            seen.add(rule_id)
            unique_constraints.append(rule)

    return unique_constraints



def apply_targeted_edit(edges, mutation_type, edge):
    """
    Applies ONE specific edit to an edge list.

    Used when the search procedure (e.g. LA-GHC / LANO) decides
    WHICH node relationship is offered up for mutation, while the
    mutation operator still decides HOW the mutation occurs.

    Returns
    -------
    (candidate_edges, affected_nodes), or None if the edit is not
    applicable to the current edge list.
    """

    a, b = edge
    # Changed: Normalize stored edge lists to tuples before testing membership.
    candidate_edges = [tuple(pair) for pair in edges]

    if mutation_type == "edge_add":
        # Can't add an edge that already exists
        if (a, b) in candidate_edges:
            return None
        candidate_edges.append((a, b))

    elif mutation_type == "edge_delete":
        # Can only delete an edge that exists
        if (a, b) not in candidate_edges:
            return None
        candidate_edges.remove((a, b))

    # Changed: Accept the reversal name used by the RQ2 experiment grids.
    elif mutation_type in ("edge_flip", "edge_reverse"):
        # Can only flip an edge that exists: a -> b becomes b -> a
        if (a, b) not in candidate_edges:
            return None
        i = candidate_edges.index((a, b))
        candidate_edges[i] = (b, a)

    else:
        return None

    # New edges, plus the two variables the edit touched
    return candidate_edges, {a, b}


# Changed: All three operators share a DAG and parent-limit check.
def valid_dag(nodes, edges, max_in_degree=None):
    """Check the fixed node set, simple directed edges, parent cap and acyclicity."""
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for parent, child in edges:
        if parent not in graph or child not in graph or graph.has_edge(parent, child):
            return False
        graph.add_edge(parent, child)
    if max_in_degree is not None and any(d > max_in_degree for _, d in graph.in_degree()):
        return False
    return nx.is_directed_acyclic_graph(graph)


# Changed: Retry failed random edits from the incumbent; try a targeted edge only once.
def _edited_states(state, mutation_type, max_attempts, edge):
    """Yield structurally legal edits of the incumbent, using fresh edges on each draw."""
    nodes, edges = state["nodes"], state["edges"]
    for _ in range(1 if edge is not None else max_attempts):
        if edge is not None:
            selected = edge
        elif mutation_type == "edge_add" and len(nodes) >= 2:
            selected = random.sample(nodes, 2)
        elif mutation_type in ("edge_delete", "edge_flip", "edge_reverse") and edges:
            selected = random.choice(edges)
        else:
            return

        edited = apply_targeted_edit(edges, mutation_type, selected)
        if edited is None:
            continue
        candidate_edges, affected = edited
        if not valid_dag(nodes, candidate_edges, state.get("max_in_degree")):
            continue
        candidate = copy.deepcopy(state)
        candidate["edges"] = candidate_edges
        yield candidate, affected


# Changed: Duplicate additions and invalid graphs no longer end a random search early.
def standard_mutation(lcn_state, mutation_type="edge_add", max_attempts=10, edge=None):
    """Apply the requested edge edit and return a structurally valid state.

    In sampled mode, try up to max_attempts edits of mutation_type, each
    from the original graph. A duplicate addition, a cycle or an exceeded
    parent cap can consume an attempt. Return the first valid candidate, or
    the original state if the edit is unavailable or the attempts run out.

    edge is an optional (source, target) pair used by targeted search such
    as LA-GHC. If supplied, try that specific edit once. This baseline
    uses no rule-derived edge requirements.
    """
    for candidate, _ in _edited_states(lcn_state, mutation_type, max_attempts, edge):
        return candidate
    return lcn_state


# Changed: Test structural validity before the indexed rule check on each draw.
def constraint_aware_mutation(
    lcn_state, constraint_index, mutation_type="edge_add", max_attempts=10, edge=None
):
    """Reject a single-edge edit if it violates a locally required dependency.

    Start from a feasible graph for RQ2. For each structurally valid edit,
    use constraint_index to check the rules mentioning the changed nodes.
    For example, if X2 -> X3 is required, either deleting it or replacing
    it with X3 -> X2 removes the required direction and fails the check.

    A rejected random edit is discarded; the next attempt starts from the
    incumbent. Return the first candidate that passes or the incumbent if
    none succeeds within max_attempts. When edge=(source, target) is
    supplied, only that edit is checked and there is no random retry.
    """
    for candidate, affected in _edited_states(lcn_state, mutation_type, max_attempts, edge):
        if not local_constraint_violation(candidate, affected, constraint_index):
            return candidate
    return lcn_state


def post_mutation_contraint_repair(
    lcn_state, constraint_index, mutation_type="edge_add", max_attempts=10,
    max_repairs=3, edge=None
):
    """Repair missing required edges after a standard mutation.

    1. Apply standard_mutation. The optional edge=(source, target) selects
       one specific edit; otherwise max_attempts limits the random draws.
    2. Compare the original and edited edge sets and identify the endpoints
       of the changed edges.
    3. Retrieve the rules mentioning those endpoints and extract the
       required directed pairs. If max_repairs is positive, insert all
       missing incident pairs. With fixed rules one round suffices.
    4. Validate the result. RQ2 checks all required arcs, acyclicity and
       the configured parent limit, including when max_repairs is zero.
       The legacy targeted RQ3 path without a max_in_degree state key
       checks the incident requirements and permits an incomplete prior.

    Restoring a deleted required edge reproduces the incumbent. Restoring
    a reversed required edge leaves a two-cycle, which fails validation.
    Return the incumbent itself in either case so the caller can recognise
    an unchanged graph and reuse its score.
    """
    candidate = standard_mutation(lcn_state, mutation_type, max_attempts, edge=edge)
    if candidate is lcn_state:
        return lcn_state

    original = set(map(tuple, lcn_state["edges"]))
    edges = set(candidate["edges"])
    affected = {node for pair in original ^ edges for node in pair}
    rules = lcn_state["logical_constraints"]
    incident = get_relevant_constraints(constraint_index, rules, affected)
    local_arcs = extract_constraint_dependencies(incident)
    missing = local_arcs - edges
    # Changed: A positive repair budget restores all missing incident arcs in one round.
    if missing and max_repairs > 0:
        # The rules are fixed: one round inserts all missing incident arcs.
        candidate["edges"].extend(sorted(missing))
        edges.update(missing)

    # RQ3's targeted edits can start from an incomplete prior and check incident
    # rules only. RQ2 always starts feasible and checks the full required set.
    if edge is not None and "max_in_degree" not in lcn_state:
        required = local_arcs
    else:
        required = extract_constraint_dependencies(rules)
    # Changed: Validate the repaired graph even when the repair budget is zero.
    if not required.issubset(edges) or not valid_dag(
        candidate["nodes"], candidate["edges"], lcn_state.get("max_in_degree")
    ):
        return lcn_state
    # Changed: Return the incumbent itself when repair undoes the edit so scoring can be skipped.
    return lcn_state if edges == original else candidate
