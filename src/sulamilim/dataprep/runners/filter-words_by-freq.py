import csv
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path

import fasttext


PATH = '/Users/ronp/Documents/wiki.he/wiki.he.bin'

_HEBREW_WORD_RANGE = range(1488, 1515)


def is_valid_hebrew_word(word):
    """
    Check if a word contains only Hebrew characters.
    Hebrew Unicode range: 0x0590-0x05FF
    """
    return all(map(_HEBREW_WORD_RANGE.__contains__, map(ord, word)))


def get_words_frequencies() -> dict[str, int]:
    ft = fasttext.load_model(PATH)
    words = ft.get_words(include_freq=True)
    return dict(zip(words[0], words[1]))


def load_words(file_path, word_length) -> list[str]:
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

    return list(valid_words)


def sort_words_by_freq(words: list[str], word_freqs: dict[str, int]) -> list[tuple[str, int]]:
    words_with_freqs = zip(words, map(word_freqs.get, words))
    words_with_freqs = list(filter(lambda wf: bool(wf[1]), words_with_freqs))
    sorted_word_freqs = sorted(words_with_freqs, key=itemgetter(1), reverse=True)
    return sorted_word_freqs


def write_sorted_words(path: Path, sorted_word_freqs: list[tuple[str, int]]) -> None:
    with open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'freq'])
        writer.writerows(sorted_word_freqs)


def main():
    parser = ArgumentParser()

    parser.add_argument('--words-path', '-w', type=Path, required=True, help='Path to the CSV file containing words')
    parser.add_argument('--length', '-l', type=int, required=True, help='Length of words to include in the network')
    parser.add_argument('--output-prefix', '-o', type=Path, default='sorted-word-freqs',
                        help='Output file path for the network (default: word_network.graphml)')

    args = parser.parse_args()
    output_path = Path(args.output_prefix).with_suffix('.csv')

    words = load_words(args.words_path, args.length)
    word_freqs = get_words_frequencies()
    sorted_word_freqs = sort_words_by_freq(words, word_freqs)
    write_sorted_words(output_path, sorted_word_freqs)


if __name__ == '__main__':
    main()





