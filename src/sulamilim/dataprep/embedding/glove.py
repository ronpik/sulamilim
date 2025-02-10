from typing import Dict

import numpy as np


def parse_glove_line(line: str) -> tuple[str, np.ndarray]:
    parts = line.strip().split()
    if not parts:
        raise ValueError("Could not parse line")

    word = parts[0]
    vec = np.array(parts[1:], dtype=np.float32)
    return word, vec


def load_glove_embeddings(glove_path: str, normalize: bool = True) -> Dict[str, np.ndarray]:
    words: list[str] = []
    vectors: list[np.ndarray] = []
    with open(glove_path, "r", encoding="utf8") as f:
        for line in f:
            try:
                word, vec = parse_glove_line(line)
                words.append(word)
                vectors.append(vec)
            except Exception as e:
                print(f"Error parsing line: {line}: {e}")
                continue

    # Stack all vectors into a matrix
    vec_matrix = np.vstack(vectors)
    if normalize:
        # Compute L2 norms for each vector (avoiding division by zero)
        norms = np.linalg.norm(vec_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vec_matrix = vec_matrix / norms

    # Reconstruct the dictionary mapping words to their (normalized) vectors
    embeddings = {word: vec for word, vec in zip(words, vec_matrix)}
    return embeddings
