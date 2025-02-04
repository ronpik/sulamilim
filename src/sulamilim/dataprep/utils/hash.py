import xxhash


def compute_pair_hash(word1: str, word2: str) -> str:
    """
    Compute a hash for an unordered pair of words.
    Each word is hashed using xxhash and the two integer hash values are summed.
    (Since addition is commutative, order does not matter.)
    """
    hash1 = xxhash.xxh64(word1).intdigest()
    hash2 = xxhash.xxh64(word2).intdigest()
    return str(hash1 + hash2)