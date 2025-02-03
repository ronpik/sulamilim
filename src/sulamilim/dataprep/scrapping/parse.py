from typing import Iterator, Optional

import httpx
from bs4 import BeautifulSoup


def parse_html(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, 'html.parser')
    return soup


def iter_all_links(soup: BeautifulSoup, base_url: Optional[str] = None) -> Iterator[str]:
    for link in soup.find_all("a", href=True):
        href = link["href"]
        yield href


def iter_pdf_links(soup: BeautifulSoup, base_url: Optional[str] = None) -> Iterator[str]:
    links = iter_all_links(soup)
    for link in links:
        if link.lower().endswith(".pdf"):
            yield link
