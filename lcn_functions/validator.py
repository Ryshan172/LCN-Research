 # Constraint checker 

def validate_lcn(lcn):
    """
    Validate a Logical Credal Network (LCN) structure and parameters.
    
    Checks:
        - DAG validity (no cycles)
        - Logical constraints validity
        - Credal set probability intervals validity
    
    Args:
        lcn (dict): Logical Credal Network with keys:
            "nodes", "edges", "credal_sets", "logical_constraints"
    
    Returns:
        dict: {
            "is_valid": bool,
            "errors": list of str
        }
    """
    errors = []
    
    nodes = lcn.get("nodes", [])
    edges = lcn.get("edges", [])
    credal_sets = lcn.get("credal_sets", {})
    constraints = lcn.get("logical_constraints", [])
    
    # --- 1. Check DAG validity ---
    if not _is_dag(nodes, edges):
        errors.append("Graph contains cycles, not a DAG.")
    
    # --- 2. Check constraint validity ---
    for idx, cons in enumerate(constraints):
        if_part = cons.get("if", {})
        then_part = cons.get("then", {})
        
        # Node existence
        for node in list(if_part.keys()) + list(then_part.keys()):
            if node not in nodes:
                errors.append(f"Constraint {idx} references unknown node '{node}'.")
        
        # Ancestor check (optional but good for semantics)
        for if_node in if_part:
            for then_node in then_part:
                if not _is_ancestor(if_node, then_node, edges):
                    errors.append(
                        f"Constraint {idx}: '{if_node}' is not an ancestor of '{then_node}' in the DAG."
                    )
    
    # --- 3. Check credal set intervals ---
    for node, conditions in credal_sets.items():
        for cond_key, probs in conditions.items():
            for val in ["True", "False"]:
                if val not in probs:
                    errors.append(f"Missing probability for '{val}' in {node} | {cond_key}")
                    continue
                interval = probs[val]
                if (not isinstance(interval, list) or len(interval) != 2 or
                    not all(isinstance(x, (int, float)) for x in interval)):
                    errors.append(f"Invalid interval format for {node} {val} in {cond_key}")
                else:
                    low, high = interval
                    if not (0 <= low <= high <= 1):
                        errors.append(f"Probability interval out of bounds for {node} {val} in {cond_key}")
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors
    }


def _is_dag(nodes, edges):
    """Return True if edges form a DAG over given nodes."""
    from collections import defaultdict, deque
    indegree = defaultdict(int)
    graph = defaultdict(list)
    
    for u, v in edges:
        graph[u].append(v)
        indegree[v] += 1
    
    # Kahn's algorithm
    q = deque([n for n in nodes if indegree[n] == 0])
    visited = 0
    
    while q:
        node = q.popleft()
        visited += 1
        for neigh in graph[node]:
            indegree[neigh] -= 1
            if indegree[neigh] == 0:
                q.append(neigh)
    
    return visited == len(nodes)


def _is_ancestor(src, tgt, edges):
    """Check if src is an ancestor of tgt in the DAG."""
    from collections import defaultdict, deque
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
    
    q = deque([src])
    visited = set()
    while q:
        node = q.popleft()
        if node == tgt:
            return True
        for neigh in graph[node]:
            if neigh not in visited:
                visited.add(neigh)
                q.append(neigh)
    return False