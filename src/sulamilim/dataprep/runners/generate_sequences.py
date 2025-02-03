import warnings
from argparse import ArgumentParser
from itertools import combinations
from typing import Iterator, Sequence

import networkx as nx
from tqdm import tqdm

from sulamilim.dataprep.utils.graph_itertools import enumerate_connected_subgraphs
from sulamilim.dataprep.utils.graphlets import generate_graphlets
from sulamilim.dataprep.utils.hamilton import all_hamiltonian_paths


def is_valid_graphlet(graphlet: nx.Graph):
    return len(all_hamiltonian_paths(graphlet)) == 1


def generate_valid_graphlets(n: int) -> Iterator[nx.Graph]:
    """
    Enumerate all (non-isomorphic) connected graphs on n nodes that have a unique
    Hamiltonian ordering (up to reversal).
    """
    graphlets_iter = generate_graphlets(n)
    yield from filter(is_valid_graphlet, graphlets_iter)


def find_valid_sequences(
        G: nx.Graph,
        valid_graphlets: list[nx.Graph],
        seq_size: int
) -> Iterator[Sequence[str]]:
    """
    Finds valid sequences by iterating only over connected induced subgraphs of G
    of size seq_size. For each such subgraph, check if it is isomorphic to one of the valid
    graphlets. If so, determine its (canonical) Hamiltonian ordering and add that ordering
    to the sequence list.
    """
    n_subsets = 0
    n_valid = 0

    # Enumerate only connected subgraphs of the desired size.
    for nodes_set in enumerate_connected_subgraphs(G, seq_size):
        if n_subsets % 10_000 == 0:
            print(f"Iterated {n_subsets=} / {n_valid=}")

        n_subsets += 1
        subG = G.subgraph(nodes_set)
        for graphlet in valid_graphlets:
            if not nx.is_isomorphic(subG, graphlet):
                continue

            paths = all_hamiltonian_paths(subG)
            if not paths:
                warnings.warn(f"Subgraph is isomorphic to {graphlet} but path wasn't found: {subG.nodes()}")
                continue

            n_valid += 1
            yield paths[0]
            break  # No need to check other graphlets for this subgraph.


def main():
    parser = ArgumentParser(description="Generate valid word-ladder sequences for CrossClimb.")

    parser.add_argument("--input-file", '-i', help="Path to the GraphML file for the words network.")
    parser.add_argument("--output-prefix", '-o', default='sequences', help="Path where the sequences will be saved (one sequence per line).")
    args = parser.parse_args()

    # Load the word network (each node is a word; edges connect words that differ by one letter)
    G = nx.read_graphml(args.input_file)
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    for i in [4, 5]:
        valid_graphlets = list(generate_valid_graphlets(i))
        out_path = f"{args.output_prefix}-{i}.txt"
        with open(out_path, 'w') as f:
            for seq in find_valid_sequences(G, valid_graphlets, i):
                f.write(" - ".join(seq) + "\n")


if __name__ == "__main__":
    main()