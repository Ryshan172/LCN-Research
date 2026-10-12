from mutations.mutation_functions import constraint_aware_mutation, standard_mutation, post_mutation_contraint_repair
import copy
import heapq
import random

"""
Logic-Aware Variable-Ordered GHC (LA-GHC)

An algorithm designed to make use of logic-aware variable ordering
to better navigate the search space during LCN structure learning 
"""



"""
Logic-Aware Variable-Ordered GHC (LA-GHC)

LANO:
    Logic-Aware Neighbourhood Ordering.
    Ranks candidate graph edits according to their relevance
    to the logical constraints.

LA-GHC:
    Uses LANO to guide candidate exploration, an existing
    mutation operator to apply edits, and Interval BIC to
    evaluate candidate structures.
"""


# 1. CANDIDATE GENERATION
def generate_candidate_edits(lcn_state):
    """
    Generate possible single-edge graph edits.

    Returns dictionaries describing:
        edge_add
        edge_delete
        edge_reverse

    This function proposes edits; it does not apply them.
    """
    nodes = lcn_state["nodes"]
    edges = set(map(tuple, lcn_state["edges"]))

    for source in nodes:
        for target in nodes:

            if source == target:
                continue

            edge = (source, target)
            reverse_edge = (target, source)

            # Add an edge if neither orientation exists.
            if edge not in edges and reverse_edge not in edges:
                yield {
                    "type": "edge_add",
                    "edge": edge,
                }

            # Delete or reverse an existing edge.
            if edge in edges:
                yield {
                    "type": "edge_delete",
                    "edge": edge,
                }

                yield {
                    "type": "edge_reverse",
                    "edge": edge,
                }


# 2. LOGIC-AWARE PRIORITY
def logic_aware_edit_priority(edit, constraint_index):
    """
    Estimate how relevant a candidate edit is to the
    supplied logical constraints.

    Higher priority means greater constraint relevance.

    IMPORTANT:
    Constraint relevance is not a guarantee of improved
    logical consistency or a better Interval BIC score.
    """
    source, target = edit["edge"]

    source_constraints = constraint_index.get(source, [])
    target_constraints = constraint_index.get(target, [])

    source_ids = {id(rule) for rule in source_constraints}

    shared_constraints = sum(
        id(rule) in source_ids
        for rule in target_constraints
    )

    return float(shared_constraints)


def logic_aware_priority_queue(
    lcn_state,
    tabu_moves=None,
    exploration_counts=None,
):
    """
    Rank candidate edits using LANO.

    Inputs:
        lcn_state:
            Current graph and logical constraint information.

        tabu_moves:
            Set of temporarily forbidden edit keys.

        exploration_counts:
            Number of times each edit has been considered.

    Returns:
        Candidate edits ordered from highest to lowest priority.

    The queue does not mutate the graph or calculate IBIC.
    """
    tabu_moves = tabu_moves or set()
    exploration_counts = exploration_counts or {}

    constraint_index = lcn_state.get("constraint_index", {})

    priority_queue = []
    counter = 0

    for edit in generate_candidate_edits(lcn_state):

        source, target = edit["edge"]
        edit_key = (edit["type"], source, target)

        if edit_key in tabu_moves:
            continue

        relevance = logic_aware_edit_priority(
            edit,
            constraint_index,
        )

        visits = exploration_counts.get(edit_key, 0)

        # Small exploration bonus for less frequently
        # considered edits.
        exploration_bonus = 1.0 / (1.0 + visits)

        priority = relevance + 0.25 * exploration_bonus

        # heapq is a min-heap, so negate priority.
        heapq.heappush(
            priority_queue,
            (-priority, counter, edit),
        )

        counter += 1

    ordered_edits = []

    while priority_queue:
        _, _, edit = heapq.heappop(priority_queue)
        ordered_edits.append(edit)

    return ordered_edits


# 3. MUTATION OPERATOR DISPATCH
def apply_ordered_mutation(
    lcn_state,
    candidate_edit,
    mutation_strategy="constraint",
    mutation_context=None,
):
    """
    Integration point between LANO and the existing operators.

    Required contract:
        Apply the SPECIFIC candidate_edit selected by LANO.

    The current RQ2 operator interface accepts mutation_type,
    but does not explicitly accept a selected node relationship.
    Therefore, targeted-edit support must be added to the
    operator interfaces or implemented in a compatible adapter.

    This function deliberately does not silently call a random
    mutation, since that could apply a different edit from the
    one LANO selected.
    """
    raise NotImplementedError(
        "Connect candidate_edit to the existing mutation "
        "operators while preserving their validation and repair "
        "behaviour."
    )


# 4. LA-GHC
def logic_aware_greedy_hill_climbing(
    samples_df,
    aggregate_table,
    nodes,
    logical_constraints=None,
    initial_state=None,
    mutation_strategy="constraint",
    max_iter=100,
    stagnation_limit=15,
    allow_plateau=True,
    max_plateau_moves=5,
    tabu_size=20,
    scoring="mid",
    score_function=None,
    mutation_function=None,
):
    """
    Logic-Aware Variable-Ordered Greedy Hill Climbing.

    The score_function must evaluate a graph using Interval BIC.
    The mutation_function must apply the specific edit selected
    by LANO and return a candidate state, or None if invalid.

    Returns:
        best_state:
            Highest-scoring graph found.

        trajectory:
            Accepted-state score history.

        diagnostics:
            Search statistics, including stagnation and moves.
    """
    if score_function is None:
        raise ValueError(
            "Provide score_function, e.g. a wrapper around "
            "your existing score_structure(..., scoring=scoring)."
        )

    if mutation_function is None:
        raise ValueError(
            "Provide a mutation_function that applies the "
            "specific candidate edit selected by LANO."
        )

    if stagnation_limit < 1:
        raise ValueError("stagnation_limit must be positive.")

    if max_plateau_moves < 0 or tabu_size < 0:
        raise ValueError(
            "max_plateau_moves and tabu_size cannot be negative."
        )

    if initial_state is None:
        from copy import deepcopy

        from lcn_functions.model import create_lcn

        # Initialise the learner's graph, not a new ground-truth
        # LCN. Keep this state minimal and explicit.
        from utils.lcn_initialisation import initialise_structure
        current = initialise_structure(
            nodes,
            logical_constraints=logical_constraints,
        )
    else:
        current = copy.deepcopy(initial_state)

    current_score = score_function(
        current, samples_df, aggregate_table, scoring
    )

    best_state = copy.deepcopy(current)
    best_score = current_score

    trajectory = [current_score]

    tabu_moves = set()
    tabu_structures = []
    exploration_counts = {}

    stagnation = 0
    plateau_moves = 0
    accepted_uphill = 0
    accepted_neutral = 0
    rejected_candidates = 0

    for iteration in range(max_iter):

        ordered_edits = logic_aware_priority_queue(
            current,
            tabu_moves=tabu_moves,
            exploration_counts=exploration_counts,
        )

        accepted = False
        improved = False

        for edit in ordered_edits:

            source, target = edit["edge"]
            edit_key = (edit["type"], source, target)

            exploration_counts[edit_key] = (
                exploration_counts.get(edit_key, 0) + 1
            )

            candidate = mutation_function(
                copy.deepcopy(current),
                edit,
                mutation_strategy,
            )

            if candidate is None:
                rejected_candidates += 1
                continue

            # Prevent repeated traversal of recent structures.
            candidate_signature = frozenset(
                map(tuple, candidate["edges"])
            )

            if candidate_signature in tabu_structures:
                continue

            candidate_score = score_function(
                candidate,
                samples_df,
                aggregate_table,
                scoring,
            )

            # Greedy uphill acceptance.
            if candidate_score > current_score:
                current = candidate
                current_score = candidate_score

                accepted = True
                improved = True
                accepted_uphill += 1
                plateau_moves = 0

                if candidate_score > best_score:
                    best_state = copy.deepcopy(candidate)
                    best_score = candidate_score

                break

            # Optional neutral moves for plateau traversal.
            if (
                allow_plateau
                and candidate_score == current_score
                and plateau_moves < max_plateau_moves
            ):
                current = candidate
                accepted = True
                accepted_neutral += 1
                plateau_moves += 1
                break

            rejected_candidates += 1

        # Track improvement relative to the preceding iteration.
        if improved:
            stagnation = 0
        else:
            stagnation += 1

        trajectory.append(current_score)

        tabu_structures.append(
            frozenset(map(tuple, current["edges"]))
        )

        if len(tabu_structures) > tabu_size:
            tabu_structures.pop(0)

        # Stagnation is a trigger, not proof of optimality.
        if stagnation >= stagnation_limit:
            break

        # No valid or available candidates means the current
        # neighbourhood cannot be explored further.
        if not ordered_edits:
            break

    diagnostics = {
        "iterations": len(trajectory) - 1,
        "best_score": best_score,
        "final_score": current_score,
        "stagnation": stagnation,
        "accepted_uphill": accepted_uphill,
        "accepted_neutral": accepted_neutral,
        "rejected_candidates": rejected_candidates,
        "termination_reason": (
            "stagnation"
            if stagnation >= stagnation_limit
            else "max_iterations_or_neighbourhood_exhaustion"
        ),
    }

    return best_state, trajectory, diagnostics


