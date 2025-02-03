from itertools import combinations
from typing import Iterator

import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
from networkx.generators.classic import path_graph


# def is_valid_graphlet(graph):
#     """
#     A graphlet (of size 4 or 5) is considered valid if every Hamiltonian
#     path (ordering of the nodes with consecutive adjacent) has the same unordered
#     pair of endpoints. This guarantees that only two nodes can serve as the sequence edges.
#     """
#     paths = all_hamiltonian_paths(graph)
#     endpoint_pairs = set()
#     for path in paths:
#         # sort the two endpoints so that [a, b] and [b, a] count as the same pair
#         endpoints = tuple(sorted([path[0], path[-1]]))
#         endpoint_pairs.add(endpoints)
#     return len(endpoint_pairs) == 1


def generate_graphlets(n: int) -> Iterator[nx.Graph]:
    """
    Enumerate all (non-isomorphic) connected graphs on n nodes that have a unique
    Hamiltonian ordering (up to reversal).
    """
    seen_graphlets = set()
    G_path = path_graph(n)
    all_possible_extra_edges = set(combinations(range(n), 2)) - set(G_path.edges())
    for num_extra_edges in range(len(all_possible_extra_edges) + 1):
        for edges in combinations(all_possible_extra_edges, num_extra_edges):
            graphlet = G_path.copy()
            graphlet.add_edges_from(edges)
            graphlet_hash = weisfeiler_lehman_graph_hash(graphlet)
            if graphlet_hash not in seen_graphlets:
                seen_graphlets.add(graphlet_hash)
                yield graphlet

