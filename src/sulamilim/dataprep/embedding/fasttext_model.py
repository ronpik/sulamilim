import io
from pathlib import Path

import fasttext
import numpy as np
from tqdm import tqdm

PATH = '/Users/ronp/Documents/wiki.he/wiki.he.bin'
VEC_PATH = '/Users/ronp/Documents/wiki.he/wiki.he.vec'



def load_fasttext_vectors(fname: Path | str) -> dict[str, np.ndarray]:
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        vec = np.array(list(map(float, tokens[1:])))
        normalized_vec = vec / np.linalg.norm(vec)
        data[tokens[0]] = normalized_vec
    return data


if __name__ == '__main__':
    ft = fasttext.load_model(PATH)
    words = ft.get_words(include_freq=True)
    print(len(words))
    m_out = ft.get_output_matrix()
    m_in = ft.get_input_matrix()
    w1 = "שלום"
    w_id1 = ft.get_word_id(w1)
    wv1 = ft.get_word_vector(w1)
    print(f"{w1=}")
    print(f"{w_id1=}")
    print(f"{wv1.shape=}")

    w2 = "להתראות"
    w_id2 = ft.get_word_id(w2)
    wv2 = ft.get_word_vector(w2)
    print(f"{w2=}")
    print(f"{w_id2=}")
    print(f"{wv2.shape=}")

    v1 = wv1 / np.linalg.norm(wv1)
    v2 = wv2 / np.linalg.norm(wv2)
    sim = np.dot(v1, v2)
    print(f"{sim=}")

    v1 = m_in[w_id1] / np.linalg.norm(m_in[w_id1])
    v2 = m_in[w_id2] / np.linalg.norm(m_in[w_id2])
    sim = np.dot(v1, v2)
    print(f"{sim=}")

    v1 = m_out[w_id1] / np.linalg.norm(m_out[w_id1])
    v2 = m_out[w_id2] / np.linalg.norm(m_out[w_id2])
    sim = np.dot(v1, v2)
    print(f"{sim=}")

    data = load_fasttext_vectors(VEC_PATH)
    wv1 = data[w1]
    wv2 = data[w2]
    v1 = wv1 / np.linalg.norm(wv1)
    v2 = wv2 / np.linalg.norm(wv2)
    sim = np.dot(v1, v2)
    print(f"{sim=}")