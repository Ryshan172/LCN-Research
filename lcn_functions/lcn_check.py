from collections import deque

"""
Additional functions for checking:
- No unconnected edges
- Ensuring that the generated LCN is acyclic 
"""

def check_all_nodes_connected(nodes, edges):
    """
    Check if all nodes are connected (no isolated components).
    Uses BFS/DFS to test connectivity.
    """
    if not edges:
        return False

    # Build adjacency list
    adj = {n: [] for n in nodes}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # BFS from first node
    visited = set()
    queue = deque([nodes[0]])
    while queue:
        curr = queue.popleft()
        if curr not in visited:
            visited.add(curr)
            queue.extend(adj[curr])

    return len(visited) == len(nodes)


def validate_generated_lcn(lcn):
    """
    Validate a Logical Credal Network (LCN) for basic structural correctness.
    Returns True if valid, False otherwise, and prints checks.
    """
    nodes = lcn.get("nodes", [])
    edges = lcn.get("edges", [])
    credal_sets = lcn.get("credal_sets", {})
    logical_constraints = lcn.get("logical_constraints", [])

    valid = True

    # Check DAG acyclicity
    from collections import defaultdict, deque

    def is_acyclic(nodes, edges):
        # Build adjacency list
        adj = defaultdict(list)
        in_deg = {n: 0 for n in nodes}
        for p, c in edges:
            adj[p].append(c)
            in_deg[c] += 1
        # Kahn's algorithm
        queue = deque([n for n in nodes if in_deg[n] == 0])
        visited = 0
        while queue:
            n = queue.popleft()
            visited += 1
            for child in adj[n]:
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)
        return visited == len(nodes)

    if not is_acyclic(nodes, edges):
        print("DAG contains cycles.")
        valid = False
    else:
        print("DAG is acyclic.")

    # Check CPT counts per node
    for node in nodes:
        # Find parents
        parents = [p for p, c in edges if c == node]
        expected_cpts = 2 ** len(parents)
        actual_cpts = len(credal_sets.get(node, {}))
        if actual_cpts != expected_cpts:
            print(f"Node {node} has {actual_cpts} CPTs, expected {expected_cpts}.")
            valid = False
        else:
            print(f"Node {node} CPT count is correct ({actual_cpts}).")

    # Check logical constraints consistency
    for constr in logical_constraints:
        if_node, if_val = list(constr["if"].items())[0]
        then_node, then_val = list(constr["then"].items())[0]
        # Find CPT for matching parent assignment
        if then_node in credal_sets:
            cpts = credal_sets[then_node]
            matched = False
            for cond_key, probs in cpts.items():
                if if_node in cond_key:
                    # Check if the constraint is strictly applied
                    if then_val and probs["True"][0] == 1.0 and probs["False"][1] == 0.0:
                        matched = True
                    elif not then_val and probs["True"][1] == 0.0 and probs["False"][0] == 1.0:
                        matched = True
            if not matched:
                print(f"Constraint {if_node} → {then_node} may not be reflected in CPTs.")
                valid = False
            else:
                print(f"Constraint {if_node} → {then_node} correctly applied.")

    return valid
