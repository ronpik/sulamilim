import networkx as nx


def all_hamiltonian_paths(graph: nx.Graph):
    """
    Return all Hamiltonian paths (each as a list of nodes) in the given graph.
    (For 4–5 node graphs, a brute‐force DFS is acceptable.)
    """
    paths = []
    nodes = list(graph.nodes())
    n = len(nodes)

    def dfs(path, visited):
        if len(path) == n:
            paths.append(path.copy())
            return
        for neighbor in graph.neighbors(path[-1]):
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(path, visited)
                path.pop()
                visited.remove(neighbor)

    for start in nodes:
        dfs([start], {start})

    unique_paths = list(set(tuple(reversed(path)) for path in paths if path[0] > path[-1]))
    return unique_paths