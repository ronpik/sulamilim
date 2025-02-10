import networkx as nx
from typing import Iterator, Set, FrozenSet


def enumerate_connected_subgraphs(G: nx.Graph, k: int) -> Iterator[FrozenSet]:
    """
    Yields all connected induced subgraphs of G with exactly k nodes,
    represented as frozensets of nodes.

    To avoid duplicates, for each connected subgraph the smallest node (in sorted order)
    is enforced to be the starting seed.
    """
    nodes = sorted(G.nodes())
    for seed in nodes:
        initial = {seed}
        # Only consider neighbors with a larger label than the seed
        candidates = {n for n in G.neighbors(seed) if n > seed}
        yield from _extend_connected_subgraph(G, initial, candidates, seed, k)


def _extend_connected_subgraph(G: nx.Graph, subgraph: Set, candidates: Set, seed, k: int) -> Iterator[FrozenSet]:
    """
    Helper function for recursively extending the subgraph.
    """
    if len(subgraph) == k:
        yield frozenset(subgraph)
    else:
        # Process candidates in sorted order to enforce consistency
        for v in sorted(candidates):
            new_subgraph = subgraph | {v}
            # New candidates are the union of the current candidates and v's neighbors,
            # minus those already in the subgraph.
            new_candidates = (candidates | set(G.neighbors(v))) - new_subgraph
            # Enforce that the seed remains the smallest element.
            if min(new_subgraph) == seed:
                yield from _extend_connected_subgraph(G, new_subgraph, new_candidates, seed, k)
