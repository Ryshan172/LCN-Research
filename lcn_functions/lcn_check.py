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
