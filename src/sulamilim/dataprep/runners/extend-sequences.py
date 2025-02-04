import argparse
import csv
import itertools
from typing import Dict

import networkx as nx
import numpy as np

from sulamilim.dataprep.embedding.glove import load_glove_embeddings
from sulamilim.dataprep.utils.hash import compute_pair_hash


def process_sequences(
    seq_file: str,
    G: nx.Graph,
    embeddings: Dict[str, np.ndarray],
    output_csv: str
):
    """
    Reads the sequences from the input file (each line with words separated by ' - '),
    obtains the neighborhoods for the left- and right-edge words,
    builds embedding matrices for the neighbors, computes the dot product similarity
    between all pairs at once, and writes the results to a CSV file.
    """
    results = []
    with open(seq_file, encoding="utf8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Split the sequence by the separator " - "
            words = [w.strip() for w in line.split(" - ")]
            if len(words) < 2:
                continue

            # Get the left- and right-edge words
            left_word, right_word = words[0], words[-1]

            # Get the neighbors from the word network (if the word is not in the graph, skip)
            if left_word not in G or right_word not in G:
                continue
            left_neighbors = set(G.neighbors(left_word))
            right_neighbors = set(G.neighbors(right_word))
            # Optionally, remove the word itself if present
            left_neighbors.discard(left_word)
            right_neighbors.discard(right_word)

            # Filter neighbors to those that have embeddings available.
            left_neighbors_filtered = [w for w in left_neighbors if w in embeddings]
            right_neighbors_filtered = [w for w in right_neighbors if w in embeddings]

            # Skip if either list is empty.
            if not left_neighbors_filtered or not right_neighbors_filtered:
                continue

            # Build embedding matrices for left and right neighbors.
            left_embeds = np.array([embeddings[w] for w in left_neighbors_filtered])
            right_embeds = np.array([embeddings[w] for w in right_neighbors_filtered])
            # Compute the similarity matrix (dot product, since embeddings are normalized)
            sim_matrix = left_embeds.dot(right_embeds.T)

            # Iterate over all pairs of indices using itertools.product
            for i, j in itertools.product(range(len(left_neighbors_filtered)), range(len(right_neighbors_filtered))):
                similarity = float(sim_matrix[i, j])
                neighbor_left = left_neighbors_filtered[i]
                neighbor_right = right_neighbors_filtered[j]
                pair_hash = compute_pair_hash(neighbor_left, neighbor_right)
                results.append({
                    "pair_hash": pair_hash,
                    "word1": neighbor_left,
                    "word2": neighbor_right,
                    "sequence-index": idx,
                    "similarity": similarity
                })

    # Write the results to CSV
    fieldnames = ["pair_hash", "word1", "word2", "sequence-index", "similarity"]
    with open(output_csv, "w", newline="", encoding="utf8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Results written to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract neighbor pairs from valid sequences and compute cosine similarity using normalized GloVe embeddings."
    )
    parser.add_argument("input_file", help="Path to the input file with valid sequences (one per line, words separated by ' - ').")
    parser.add_argument("--output_prefix", default="output", help="Prefix for the output CSV file (default: 'output').")
    parser.add_argument("--graph", required=True, help="Path to the word network (GraphML file).")
    parser.add_argument("--glove", required=True, help="Path to the GloVe embeddings file (text format).")
    args = parser.parse_args()

    # Load the word network
    print("Loading word network...")
    G = nx.read_graphml(args.graph)

    # Load and normalize GloVe embeddings
    print("Loading GloVe embeddings...")
    embeddings = load_glove_embeddings(args.glove)
    print(f"Loaded {len(embeddings)} embeddings.")

    # Determine output CSV file name
    output_csv = f"{args.output_prefix}_pairs.csv"

    # Process sequences to extract neighbor pairs and compute cosine similarities
    process_sequences(args.input_file, G, embeddings, output_csv)

if __name__ == "__main__":
    main()
