import argparse
import csv
from pathlib import Path

import networkx as nx

from tqdm import tqdm

_HEBREW_WORD_RANGE = range(1488, 1515)


def is_valid_hebrew_word(word):
    """
    Check if a word contains only Hebrew characters.
    Hebrew Unicode range: 0x0590-0x05FF
    """
    return all(map(_HEBREW_WORD_RANGE.__contains__, map(ord, word)))


def hamming_distance(word1, word2):
    """
    Calculate the Hamming distance between two words.
    Returns the number of positions at which corresponding characters differ.
    """
    if len(word1) != len(word2):
        return float('inf')
    return sum(c1 != c2 for c1, c2 in zip(word1, word2))


def load_words(file_path, word_length):
    """
    Load words from CSV file and filter based on length and valid characters.

    Args:
        file_path (str): Path to the CSV file
        word_length (int): Required length of words to include

    Returns:
        set: Set of valid words
    """
    valid_words = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row['word'].strip()
            if len(word) == word_length:
                if is_valid_hebrew_word(word):
                    valid_words.add(word)

    return valid_words


def build_word_network(words):
    """
    Build a network where words are connected if they differ by one character.

    Args:
        words (set): Set of words to include in the network

    Returns:
        networkx.Graph: Graph representing the word network
    """
    G = nx.Graph()
    G.add_nodes_from(words)

    # Compare each pair of words
    word_list = list(words)
    for i in tqdm(range(len(word_list))):
        for j in range(i + 1, len(word_list)):
            if hamming_distance(word_list[i], word_list[j]) == 1:
                G.add_edge(word_list[i], word_list[j])

    return G


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Build a word network from CSV file')
    parser.add_argument('--words-path', '-w', type= Path, required=True, help='Path to the CSV file containing words')
    parser.add_argument('--length', '-l', type=int, required=True, help='Length of words to include in the network')
    parser.add_argument('--output-prefix', '-o',  default='word_network',
                        help='Output file path for the network (default: word_network.graphml)')

    args = parser.parse_args()

    # Load and filter words
    words = load_words(args.words_path, args.length)
    print(f"Loaded {len(words)} valid words of length {args.length}")

    # Build the network
    G = build_word_network(words)
    print(f"Network built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    output_prefix = args.output_prefix
    output = f"{output_prefix}-{args.length}.graphml"

    # Save the network
    nx.write_graphml(G, output)
    print(f"Network saved to {output}")

    # Print some basic network statistics
    print("\nNetwork Statistics:")
    print(f"Number of connected components: {nx.number_connected_components(G)}")

    if G.number_of_nodes() > 0:
        print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")

        # Find the largest component
        largest_cc = max(nx.connected_components(G), key=len)
        print(f"Largest component size: {len(largest_cc)}")


if __name__ == "__main__":
    main()