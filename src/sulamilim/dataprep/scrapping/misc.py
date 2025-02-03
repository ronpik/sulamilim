from itertools import groupby
from operator import itemgetter
from typing import Any, Iterator, Iterable


def validate_initial(s: str, initial: str) -> str:
    if not s.startswith(initial):
        s = initial + s

    return s


def get_unique(items: Iterable[str]) -> Iterator[str]:
    yield from map(itemgetter(0), groupby(items))


if __name__ == '__main__':
    ss = "this/is/a/path"
    ss2 = "/this/is/a/path"
    ssi1 = validate_initial(ss, '/')
    ssi2 = validate_initial(ss2, '/')

    assert ssi1 == ssi2