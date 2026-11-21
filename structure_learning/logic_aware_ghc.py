import copy
import random
from collections import deque

import networkx as nx

from mutations.mutation_functions import (
    constraint_aware_mutation,
    extract_constraint_dependencies,
    post_mutation_contraint_repair,
    standard_mutation,
)


"""
Logic-Aware Variable-Ordered GHC (LA-GHC)

An algorithm designed to make use of logic-aware variable ordering
to better navigate the search space during LCN structure learning.

Division of responsibility
--------------------------
    LANO (Logic-Aware Neighbourhood Ordering)
        decides WHAT is offered up for mutation, i.e. the order in
        which candidate node relationships (graph edits) are explored.

    Mutation operators (standard / constraint / repair)
        decide HOW a mutation occurs. They are the unchanged RQ2
        operators, called with the specific edit selected by LANO.

    Interval BIC
        decides whether the mutated structure is accepted. The scoring
        function is identical to RQ2 and is injected by the caller.

The contribution is therefore confined to how the search space is
traversed. Scoring and mutations are held fixed.


Why ordering matters (and where it is placed in GHC)
----------------------------------------------------
Textbook GHC evaluates the WHOLE neighbourhood and moves to the best
neighbour. Under that rule the order in which neighbours are generated
has no effect on the result, so an ordering heuristic cannot contribute.

LA-GHC instead uses FIRST-IMPROVEMENT hill climbing (as RQ2 did, by
sampling one neighbour at a time): neighbours are evaluated in priority
order and the first acceptable one is taken. Under a finite evaluation
budget the ordering then determines
    (a) which improving move is taken when several exist, and
    (b) how many evaluations are spent before an improving move is found.
Both directly affect the structure that is reached.

The baseline used for comparison (ordering="random") is the same
procedure with a uniformly shuffled neighbourhood, i.e. stochastic
first-improvement GHC with tabu list and random restarts.


Escaping local optima
---------------------
Stagnation does not imply global optimality. The search uses explicit
diversification policies, kept separate from LANO:

    Plateau traversal:
        equal-score (neutral) moves are accepted for a limited number of
        consecutive steps. This matters for Interval BIC because adding a
        parent that is absent from the aggregate table leaves the score
        unchanged, so the landscape contains large plateaus.

    Tabu list:
        (i) the inverse of recently applied moves is tabu, unless the move
            yields a new best score (aspiration criterion);
        (ii) recently visited structures are tabu, preventing cycling.

    Random restart:
        triggered when the neighbourhood contains no acceptable move
        (local optimum / end of plateau) or no new best score has been
        found for `stagnation_limit` iterations. The search restarts from
        the best structure perturbed by random edits (iterated local
        search), so restarts do not simply retrace the previous path.

Limits
------
Every non-improving mechanism is bounded:

    max_plateau_moves         consecutive neutral moves
    max_total_plateau_moves   neutral moves per restart cycle
    stagnation_limit          iterations without a new best score
    tabu_size                 tenure of tabu moves and structures
    max_restarts              number of restarts
    perturbation_size         random edits applied at each restart
    max_evaluations           score evaluations (hard stop)

Limits left as None are scaled to the number of variables n (see
resolve_search_limits), since a fixed limit is generous for small
networks and restrictive for large ones.
"""


# Search-level edit names -> mutation operator names
EDIT_TO_OPERATOR = {
    "edge_add": "edge_add",
    "edge_delete": "edge_delete",
    "edge_reverse": "edge_flip",
}

# Weights for the logic-aware priority function
DEFAULT_LANO_WEIGHTS = {
    # Variable-level features (logic-aware variable ordering)
    "constraint_count": 1.0,
    "constraint_type": 1.0,
    "relationships": 1.0,
    "interaction": 0.5,

    # Edit-level features
    "pair_link": 2.0,
    "violation_change": 3.0,

    # Small bonus for less frequently considered edits
    "exploration": 0.1,
}

# Constraint type weights. A derived (chained) constraint is entailed
# by two others, so the dependency it expresses can be mediated by an
# intermediate variable and does not require a direct edge.
PRIMITIVE_CONSTRAINT_WEIGHT = 1.0
DERIVED_CONSTRAINT_WEIGHT = 0.5

SCORE_TOLERANCE = 1e-9


# 1. LOGIC PROFILE (static analysis of the constraint set)
def is_derived_constraint(rule, logical_constraints):
    """
    A constraint  a=va -> c=vc  is derived if some intermediate b
    exists with  a=va -> b=vb  and  b=vb -> c=vc  in the set.

    This mirrors propagate_constraints() in the LCN generator.
    Constraints with multiple antecedents/consequents are treated
    as primitive.
    """
    # Only single-variable rules (A -> C) can be chained
    if len(rule["if"]) != 1 or len(rule["then"]) != 1:
        return False

    # Unpack the rule into  a=va -> c=vc
    (a, va), = rule["if"].items()
    (c, vc), = rule["then"].items()

    # Look for a first rule that starts the same way:  a=va -> b=vb
    for first in logical_constraints:
        if first is rule or first["if"] != {a: va}:
            continue

        (b, vb), = first["then"].items()

        # b must be a different, intermediate variable
        if b in (a, c):
            continue

        # ... and a second rule that finishes the chain:  b=vb -> c=vc
        for second in logical_constraints:
            if second["if"] == {b: vb} and second["then"] == {c: vc}:
                return True

    return False


def build_logic_profile(nodes, logical_constraints, weights=None):
    """
    Precompute the logic-aware information used by LANO.

    The constraint set does not change during search, so this is
    computed once per run.

    Returns a dict with:
        pair_weight:
            (source, target) -> constraint type weight of the strongest
            constraint linking source (antecedent) to target (consequent)
        variable_features:
            node -> raw feature values
        variable_priority:
            node -> logic-aware variable priority
        variable_order:
            nodes sorted from highest to lowest priority
    """
    logical_constraints = logical_constraints or []
    weights = weights or DEFAULT_LANO_WEIGHTS

    # Give every rule a weight: 1.0 if primitive, 0.5 if derived (chained)
    rule_weights = [
        DERIVED_CONSTRAINT_WEIGHT
        if is_derived_constraint(rule, logical_constraints)
        else PRIMITIVE_CONSTRAINT_WEIGHT
        for rule in logical_constraints
    ]

    # For every (if-variable, then-variable) pair, keep the strongest
    # rule weight linking them, e.g. ("X1", "X3") -> 1.0
    pair_weight = {}

    for rule, weight in zip(logical_constraints, rule_weights):
        for source in rule["if"]:
            for target in rule["then"]:
                if source == target:
                    continue
                pair_weight[(source, target)] = max(
                    pair_weight.get((source, target), 0.0),
                    weight,
                )

    # Undirected constraint graph, used for variable interaction
    constraint_graph = nx.Graph()
    constraint_graph.add_nodes_from(nodes)
    constraint_graph.add_edges_from(pair_weight.keys())

    # Raw feature values for each variable
    raw = {}

    for node in nodes:
        # Weights of all rules that mention this variable
        involved = [
            weight
            for rule, weight in zip(logical_constraints, rule_weights)
            if node in rule["if"] or node in rule["then"]
        ]

        # Other variables this variable shares a constraint with
        counterparts = {
            other
            for pair in pair_weight
            if node in pair
            for other in pair
            if other != node
        }

        raw[node] = {
            # number of logical constraints involving the variable
            "constraint_count": len(involved),

            # type of constraints involving it
            "constraint_type": sum(involved),

            # number of relationships affected by those constraints
            "relationships": len(counterparts),

            # interaction with other constrained variables
            "interaction": len(
                nx.node_connected_component(constraint_graph, node)
            ) - 1 if counterparts else 0,
        }

    # Normalise each feature to [0, 1] so the weights are comparable
    variable_priority = {}

    for node in nodes:
        priority = 0.0

        # Weighted sum of the normalised features
        for feature, value in raw[node].items():
            max_value = max(raw[n][feature] for n in nodes) or 1
            priority += weights[feature] * value / max_value

        variable_priority[node] = priority

    # Rescale so the most important variable has priority 1.0
    max_priority = max(variable_priority.values(), default=0.0) or 1.0
    variable_priority = {
        node: priority / max_priority
        for node, priority in variable_priority.items()
    }

    # The logic-aware variable ordering (most important first)
    variable_order = sorted(
        nodes,
        key=lambda node: variable_priority[node],
        reverse=True,
    )

    return {
        "pair_weight": pair_weight,
        # Pairs X -> Y that the constraints say should be edges
        "required_pairs": extract_constraint_dependencies(
            logical_constraints
        ),
        "variable_features": raw,
        "variable_priority": variable_priority,
        "variable_order": variable_order,
    }


def logic_aware_variable_ordering(nodes, logical_constraints):
    """
    Logic-aware variable ordering: variables ranked by how strongly
    they are involved in the logical constraints.

    Returns a list of (node, priority) from highest to lowest.
    """
    profile = build_logic_profile(nodes, logical_constraints)

    return [
        (node, profile["variable_priority"][node])
        for node in profile["variable_order"]
    ]


# 2. CANDIDATE GENERATION
def creates_cycle(edges, nodes):
    """
    True if the edge list contains a directed cycle.
    """
    # Build the directed graph and ask networkx if it is a DAG
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    return not nx.is_directed_acyclic_graph(graph)


def generate_candidate_edits(lcn_state):
    """
    Generate legal single-edge graph edits.

    Legal means the edit keeps the graph acyclic (standard GHC rule):
        edge_add      (u, v): neither orientation exists and v cannot
                              already reach u
        edge_delete   (u, v): u -> v exists
        edge_reverse  (u, v): u -> v exists and u -> v is the only
                              directed path from u to v

    This function proposes edits; it does not apply them.
    """
    nodes = lcn_state["nodes"]
    edges = set(map(tuple, lcn_state["edges"]))

    # Current graph, used to check for paths (i.e. possible cycles)
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    # Consider every ordered pair of variables
    for source in nodes:
        for target in nodes:

            if source == target:
                continue

            edge = (source, target)

            if edge in edges:
                # An existing edge can always be deleted
                yield {"type": "edge_delete", "edge": edge}

                # Reversing is only safe if there is no other route
                # from source to target (otherwise it makes a cycle)
                graph.remove_edge(source, target)
                reversible = not nx.has_path(graph, source, target)
                graph.add_edge(source, target)

                if reversible:
                    yield {"type": "edge_reverse", "edge": edge}

            elif (target, source) not in edges:
                # Adding source -> target is only safe if target
                # cannot already reach source
                if not nx.has_path(graph, target, source):
                    yield {"type": "edge_add", "edge": edge}


def edit_key(edit):
    # A hashable id for an edit, e.g. ("edge_add", "X1", "X2")
    source, target = edit["edge"]
    return (edit["type"], source, target)


def inverse_edit_key(edit):
    """
    The edit that would undo `edit`. Used for the tabu list.
    """
    source, target = edit["edge"]

    # Adding is undone by deleting, and vice versa
    if edit["type"] == "edge_add":
        return ("edge_delete", source, target)

    if edit["type"] == "edge_delete":
        return ("edge_add", source, target)

    # Reversing X -> Y is undone by reversing Y -> X
    return ("edge_reverse", target, source)


# 3. LOGIC-AWARE PRIORITY
def violation_change(edit, edge_set, profile):
    """
    Change in weighted structural violations caused by the edit.

    Uses the same structural interpretation of feasibility as the
    constraint-aware mutations: a constraint X -> Y is satisfied
    structurally when the directed edge X -> Y is present.

    Negative values mean the edit removes violations, positive values
    mean it creates them. Only the (source, target) pair changes, so
    this is computed locally.
    """
    source, target = edit["edge"]
    pair_weight = profile["pair_weight"]
    required = profile["required_pairs"]

    # How much a missing required pair counts as a violation
    def weight(pair):
        return pair_weight.get(pair, PRIMITIVE_CONSTRAINT_WEIGHT)

    forward = (source, target)
    backward = (target, source)

    change = 0.0

    if edit["type"] == "edge_add":
        # Adding a required edge fixes a violation
        if forward in required:
            change -= weight(forward)

    elif edit["type"] == "edge_delete":
        # Deleting a required edge creates a violation
        if forward in required:
            change += weight(forward)

    elif edit["type"] == "edge_reverse":
        # Reversing removes source -> target and adds target -> source
        if forward in required:
            change += weight(forward)
        if backward in required:
            change -= weight(backward)

    return change


def logic_aware_edit_priority(
    edit,
    edge_set,
    profile,
    exploration_count=0,
    weights=None,
):
    """
    Logic-aware priority of a candidate edit. Higher is explored first.

        priority(e) = mean variable priority of the endpoints
                    + w_pair      * type weight of a constraint linking the pair
                    - w_violation * change in weighted structural violations
                    + w_explore   / (1 + times the edit has been considered)

    IMPORTANT:
    Constraint relevance is not a guarantee of improved logical
    consistency or a better Interval BIC score. It only determines
    the order in which edits are evaluated.
    """
    weights = weights or DEFAULT_LANO_WEIGHTS

    source, target = edit["edge"]
    variable_priority = profile["variable_priority"]
    pair_weight = profile["pair_weight"]

    # How important the two variables are (logic-aware variable ordering)
    variable_term = (
        variable_priority.get(source, 0.0)
        + variable_priority.get(target, 0.0)
    ) / 2

    # Whether a constraint directly links these two variables
    pair_term = max(
        pair_weight.get((source, target), 0.0),
        pair_weight.get((target, source), 0.0),
    )

    # Combine: important variables, linked pairs and edits that fix
    # violations rank higher; rarely tried edits get a small bonus
    return (
        variable_term
        + weights["pair_link"] * pair_term
        - weights["violation_change"] * violation_change(
            edit, edge_set, profile
        )
        + weights["exploration"] / (1.0 + exploration_count)
    )


def logic_aware_priority_queue(
    lcn_state,
    profile,
    exploration_counts=None,
    ordering="lano",
    rng=None,
    weights=None,
):
    """
    Rank candidate edits.

    ordering="lano":
        edits sorted by logic-aware priority. Ties are broken randomly,
        so edits with no logical relevance are explored in random order.

    ordering="random":
        uniformly shuffled neighbourhood (baseline GHC).

    The queue does not mutate the graph or calculate IBIC.
    """
    exploration_counts = exploration_counts or {}
    rng = rng or random.Random()

    # All legal edits from the current graph
    edits = list(generate_candidate_edits(lcn_state))

    # Baseline: try edits in a random order
    if ordering == "random":
        rng.shuffle(edits)
        return edits

    if ordering != "lano":
        raise ValueError("ordering must be 'lano' or 'random'")

    edge_set = set(map(tuple, lcn_state["edges"]))

    # Score every edit; the random number breaks ties between
    # edits with equal priority
    ranked = [
        (
            logic_aware_edit_priority(
                edit,
                edge_set,
                profile,
                exploration_counts.get(edit_key(edit), 0),
                weights,
            ),
            rng.random(),
            edit,
        )
        for edit in edits
    ]

    # Highest priority first
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)

    return [edit for _, _, edit in ranked]


# 4. MUTATION OPERATOR DISPATCH
def apply_ordered_mutation(lcn_state, candidate_edit, mutation_strategy="constraint"):
    """
    Apply the SPECIFIC candidate_edit selected by LANO using one of the
    existing mutation operators.

    Returns the candidate state, or None if the operator rejected the
    edit (the state was returned unchanged).
    """
    # Translate the edit name to the operator's name (reverse -> flip)
    mutation_type = EDIT_TO_OPERATOR[candidate_edit["type"]]
    edge = candidate_edit["edge"]
    constraint_index = lcn_state.get("constraint_index", {})

    # Hand the chosen edge to the existing RQ2 mutation operator
    if mutation_strategy == "standard":
        candidate = standard_mutation(
            lcn_state, mutation_type, edge=edge
        )

    elif mutation_strategy == "constraint":
        candidate = constraint_aware_mutation(
            lcn_state, constraint_index, mutation_type, edge=edge
        )

    elif mutation_strategy == "repair":
        candidate = post_mutation_contraint_repair(
            lcn_state, constraint_index, mutation_type, edge=edge
        )

    else:
        raise ValueError(f"Unknown strategy: {mutation_strategy}")

    # The operator returns the state unchanged when it rejects an edit
    if candidate is lcn_state or set(candidate["edges"]) == set(lcn_state["edges"]):
        return None

    return candidate


# 5. DIVERSIFICATION
def structure_signature(lcn_state):
    # A hashable fingerprint of the graph (its set of edges)
    return frozenset(map(tuple, lcn_state["edges"]))


def perturb_structure(lcn_state, num_edits, mutation_strategy, rng):
    """
    Apply `num_edits` random legal edits (through the same mutation
    operator) to diversify the search after stagnation.
    """
    state = copy.deepcopy(lcn_state)

    for _ in range(num_edits):
        # Pick legal edits in random order ...
        edits = list(generate_candidate_edits(state))
        rng.shuffle(edits)

        # ... and apply the first one the operator accepts
        for edit in edits:
            candidate = apply_ordered_mutation(state, edit, mutation_strategy)

            if candidate is None or creates_cycle(candidate["edges"], state["nodes"]):
                continue

            state = candidate
            break

    return state


def random_initial_structure(initial_state, rng, expected_edges=None):
    """
    Random DAG initialisation.

    Starting from a random structure (rather than the empty graph)
    forces the search to delete and reverse edges as well as add them,
    which makes local optima more likely.

    A random variable order is drawn and each forward pair is added
    with probability p, chosen so the expected number of edges is
    `expected_edges` (default: n, i.e. a sparse graph). Following the
    order guarantees acyclicity.

    The graph is built directly, not through a mutation operator,
    since initialisation is not a mutation.
    """
    state = copy.deepcopy(initial_state)
    nodes = list(state["nodes"])
    n = len(nodes)

    if n < 2:
        state["edges"] = []
        return state

    if expected_edges is None:
        expected_edges = n

    # Probability of each possible edge, so we get about expected_edges
    edge_prob = min(1.0, expected_edges / (n * (n - 1) / 2))

    # Random variable order; edges only go "forward" in this order,
    # so no cycle can be created
    order = nodes[:]
    rng.shuffle(order)

    state["edges"] = [
        (order[i], order[j])
        for i in range(n)
        for j in range(i + 1, n)
        if rng.random() < edge_prob
    ]

    return state


def resolve_search_limits(
    num_nodes,
    stagnation_limit=None,
    max_plateau_moves=None,
    max_total_plateau_moves=None,
    tabu_size=None,
    perturbation_size=None,
):
    """
    Fill limits left as None with values scaled to the number of
    variables n.
    """
    n = max(num_nodes, 1)

    # Defaults grow with the number of variables n
    limits = {
        "stagnation_limit": 2 * n if stagnation_limit is None else stagnation_limit,
        "max_plateau_moves": n if max_plateau_moves is None else max_plateau_moves,
        "max_total_plateau_moves": (
            3 * n if max_total_plateau_moves is None else max_total_plateau_moves
        ),
        "tabu_size": 2 * n if tabu_size is None else tabu_size,
        "perturbation_size": (
            max(2, n // 4) if perturbation_size is None else perturbation_size
        ),
    }

    if limits["stagnation_limit"] < 1:
        raise ValueError("stagnation_limit must be positive.")

    if any(
        limits[name] < 0
        for name in ("max_plateau_moves", "max_total_plateau_moves", "tabu_size", "perturbation_size")
    ):
        raise ValueError("Search limits cannot be negative.")

    return limits


# 6. LA-GHC
def logic_aware_greedy_hill_climbing(
    samples_df,
    aggregate_table,
    initial_state,
    score_function,
    mutation_strategy="constraint",
    ordering="lano",
    max_evaluations=500,
    max_iter=10_000,
    stagnation_limit=None,
    allow_plateau=True,
    max_plateau_moves=None,
    max_total_plateau_moves=None,
    tabu_size=None,
    max_restarts=5,
    perturbation_size=None,
    scoring="mid",
    lano_weights=None,
    seed=None,
):
    """
    Logic-Aware Variable-Ordered Greedy Hill Climbing.

    initial_state:
        Learner's starting state (nodes, edges, logical_constraints,
        constraint_index), e.g. from initialise_structure() in RQ2.

    score_function:
        score_function(state, samples_df, aggregate_table, scoring)
        returning Interval BIC (e.g. score_structure from RQ2).

    ordering:
        "lano" for LA-GHC, "random" for the baseline GHC with the same
        tabu, plateau and restart policies.

    max_evaluations:
        Budget of score_function calls. This is the main stopping
        criterion and makes comparisons between procedures fair.

    Limits left as None are scaled to the number of variables
    (see resolve_search_limits).

    Returns:
        best_state:
            Highest-scoring graph found.

        trajectory:
            Current score after every score evaluation.

        diagnostics:
            Search statistics, including stagnation and moves.
    """
    if max_restarts < 0:
        raise ValueError("max_restarts cannot be negative.")

    # Own random generator, so runs are reproducible with a seed
    rng = random.Random(seed)

    # Current graph S (edges stored as tuples)
    current = copy.deepcopy(initial_state)
    current["edges"] = [tuple(edge) for edge in current["edges"]]
    nodes = current["nodes"]

    # Work out all search limits (scaled to network size if not given)
    limits = resolve_search_limits(
        len(nodes),
        stagnation_limit=stagnation_limit,
        max_plateau_moves=max_plateau_moves,
        max_total_plateau_moves=max_total_plateau_moves,
        tabu_size=tabu_size,
        perturbation_size=perturbation_size,
    )
    stagnation_limit = limits["stagnation_limit"]
    max_plateau_moves = limits["max_plateau_moves"]
    max_total_plateau_moves = limits["max_total_plateau_moves"]
    tabu_size = limits["tabu_size"]
    perturbation_size = limits["perturbation_size"]

    # Analyse the constraints once (they don't change during search)
    profile = build_logic_profile(
        nodes, current.get("logical_constraints", []), lano_weights
    )

    evaluations = 0
    trajectory = []

    # Score a graph with IBIC and count it against the budget
    def evaluate(state):
        nonlocal evaluations
        evaluations += 1
        return score_function(state, samples_df, aggregate_table, scoring)

    # Evaluate IBIC(S) and set S_best = S
    current_score = evaluate(current)
    trajectory.append(current_score)

    best_state = copy.deepcopy(current)
    best_score = current_score
    best_found_at = evaluations

    # Tabu lists: oldest entries drop off automatically (maxlen)
    tabu_moves = deque(maxlen=tabu_size)
    tabu_structures = deque(maxlen=tabu_size)

    # How many times each edit has been tried (for the exploration bonus)
    exploration_counts = {}

    # Counters for limits and diagnostics
    stagnation = 0
    plateau_moves = 0
    cycle_plateau_moves = 0
    restarts = 0
    accepted_uphill = 0
    accepted_neutral = 0
    rejected_candidates = 0
    rejected_by_operator = 0
    rejected_cyclic = 0
    rejected_tabu = 0
    termination_reason = "max_iterations"

    for iteration in range(max_iter):

        # Stop when the evaluation budget is used up
        if evaluations >= max_evaluations:
            termination_reason = "evaluation_budget"
            break

        # Generate and rank candidate edits (LANO or random order)
        ordered_edits = logic_aware_priority_queue(
            current,
            profile,
            exploration_counts=exploration_counts,
            ordering=ordering,
            rng=rng,
            weights=lano_weights,
        )

        current_signature = structure_signature(current)
        moved = False
        new_best = False

        # Try edits in ranked order; take the first acceptable one
        for edit in ordered_edits:

            if evaluations >= max_evaluations:
                break

            # Remember that this edit has been tried once more
            key = edit_key(edit)
            exploration_counts[key] = exploration_counts.get(key, 0) + 1

            # Apply the mutation (standard / constraint / repair)
            candidate = apply_ordered_mutation(current, edit, mutation_strategy)

            # The operator rejected the edit
            if candidate is None:
                rejected_by_operator += 1
                continue

            # Repair may add further edges, so validity is re-checked.
            if creates_cycle(candidate["edges"], nodes):
                rejected_cyclic += 1
                continue

            candidate_signature = structure_signature(candidate)

            # A recently visited structure cannot beat the best score,
            # so it is skipped without spending an evaluation.
            if candidate_signature in tabu_structures:
                rejected_tabu += 1
                continue

            # Evaluate IBIC(S')
            candidate_score = evaluate(candidate)
            trajectory.append(current_score)

            # Tabu move: skip it, unless it beats the best score so far
            # (aspiration criterion)
            if key in tabu_moves and candidate_score <= best_score + SCORE_TOLERANCE:
                rejected_tabu += 1
                continue

            # Greedy uphill acceptance: IBIC(S') > IBIC(S)
            if candidate_score > current_score + SCORE_TOLERANCE:
                accepted_uphill += 1
                plateau_moves = 0

            # Plateau move: IBIC(S') == IBIC(S), within the plateau limits
            elif (
                allow_plateau
                and abs(candidate_score - current_score) <= SCORE_TOLERANCE
                and plateau_moves < max_plateau_moves
                and cycle_plateau_moves < max_total_plateau_moves
            ):
                accepted_neutral += 1
                plateau_moves += 1
                cycle_plateau_moves += 1

            # Worse score: reject S'
            else:
                rejected_candidates += 1
                continue

            # Accepted: forbid undoing this move, and forbid going back
            # to the graph we just left
            tabu_moves.append(inverse_edit_key(edit))
            tabu_structures.append(current_signature)

            # Move to S'
            current = candidate
            current_score = candidate_score
            trajectory[-1] = current_score
            moved = True

            # Update S_best if this is the best graph so far
            if current_score > best_score + SCORE_TOLERANCE:
                best_state = copy.deepcopy(current)
                best_score = current_score
                best_found_at = evaluations
                new_best = True

            # One move per iteration, so stop scanning
            break

        # Stagnation counts iterations without a new best score.
        stagnation = 0 if new_best else stagnation + 1

        if evaluations >= max_evaluations:
            termination_reason = "evaluation_budget"
            break

        # Stagnation is a trigger for diversification, not proof of
        # optimality.
        # Still making progress: keep climbing
        if moved and stagnation < stagnation_limit:
            continue

        # Stuck (local optimum or stagnation) and out of restarts: stop
        if restarts >= max_restarts:
            termination_reason = (
                "stagnation" if moved else "local_optimum"
            )
            break

        # Random restart: reset counters and tabu moves ...
        restarts += 1
        stagnation = 0
        plateau_moves = 0
        cycle_plateau_moves = 0
        tabu_moves.clear()

        # ... and continue from a randomly perturbed copy of S_best
        current = perturb_structure(
            best_state, perturbation_size, mutation_strategy, rng
        )
        current_score = evaluate(current)
        trajectory.append(current_score)

        if current_score > best_score + SCORE_TOLERANCE:
            best_state = copy.deepcopy(current)
            best_score = current_score
            best_found_at = evaluations

    # Summary statistics returned alongside the best graph
    edge_set = structure_signature(best_state)
    required = profile["required_pairs"]

    diagnostics = {
        "ordering": ordering,
        "iterations": iteration + 1,
        "evaluations": evaluations,
        "evaluations_to_best": best_found_at,
        "best_score": best_score,
        "final_score": current_score,
        "restarts": restarts,
        "accepted_uphill": accepted_uphill,
        "accepted_neutral": accepted_neutral,
        "rejected_candidates": rejected_candidates,
        "rejected_by_operator": rejected_by_operator,
        "rejected_cyclic": rejected_cyclic,
        "rejected_tabu": rejected_tabu,
        "termination_reason": termination_reason,
        "constraint_dependencies": len(required),
        "constraint_dependencies_satisfied": sum(
            pair in edge_set for pair in required
        ),
        "variable_order": profile["variable_order"],
        "initial_edges": len(initial_state["edges"]),
        "limits": {**limits, "max_restarts": max_restarts},
    }

    return best_state, trajectory, diagnostics
