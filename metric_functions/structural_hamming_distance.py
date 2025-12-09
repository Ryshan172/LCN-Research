def structural_hamming_distance_compare(true_edges, learned_edges):
    """
    Compute Structural Hamming Distance (SHD) between two sets of directed edges.
    Edges must be lists of tuples: (parent, child).
    """

    true_set = set(true_edges)
    learned_set = set(learned_edges)

    # Undirected versions for orientation mismatch check
    true_undirected = {frozenset(e) for e in true_set}
    learned_undirected = {frozenset(e) for e in learned_set}

    # 1. Edges that exist in true but not learned
    missing_edges = true_set - learned_set

    # 2. Edges that exist in learned but not true
    extra_edges = learned_set - true_set

    # 3. Orientation errors: same undirected edge, wrong direction
    orientation_errors = 0
    for undirected_edge in (true_undirected & learned_undirected):
        # For undirected edge {A,B}
        nodes = list(undirected_edge)
        a, b = nodes[0], nodes[1]

        if ((a, b) in true_set and (b, a) in learned_set) or \
           ((b, a) in true_set and (a, b) in learned_set):
            orientation_errors += 1

    # SHD = missing + extra + orientation errors
    shd = len(missing_edges) + len(extra_edges) + orientation_errors
    
    return {
        "shd": shd,
        "missing": list(missing_edges),
        "extra": list(extra_edges),
        "orientation_errors": orientation_errors
    }
