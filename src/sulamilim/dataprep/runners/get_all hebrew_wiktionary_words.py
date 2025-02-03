import csv
from typing import Iterator, Optional

from bs4 import BeautifulSoup
from globalog import LOG

from sulamilim.dataprep.scrapping.fetch import fetch_content
from sulamilim.dataprep.scrapping.parse import iter_all_links, parse_html

_BASE_WIKI_URL = "https://he.wiktionary.org"
_INITIAL_WORDS_SEARCH_PAGE_LOCATION = "/wiki/%D7%9E%D7%99%D7%95%D7%97%D7%93:%D7%9B%D7%9C_%D7%94%D7%93%D7%A4%D7%99%D7%9D?from=%D7%90&to=&namespace=0"
_NEXT_PAGE_TITLE = "מיוחד:כל הדפים"
_NEXT_PAGE_STR = "הדף הבא"

Html = str


def get_full_url(relative_location: str) -> str:
    return _BASE_WIKI_URL + relative_location


def get_next_url(soup: BeautifulSoup) -> Optional[str]:
    for link_tag in soup.find_all('a', title=_NEXT_PAGE_TITLE):
        if not _NEXT_PAGE_STR in link_tag.text:
            continue

        link = link_tag.get('href')
        if not link:
            continue

        return link

    return None


def iterate_words_pages() -> Iterator[tuple[str, BeautifulSoup]]:
    LOG.info("Iterating words pages")
    next_url = _INITIAL_WORDS_SEARCH_PAGE_LOCATION
    n_pages = 0
    while next_url:
        full_url = get_full_url(next_url)
        n_pages += 1
        LOG.info(f"Fetching words from page {n_pages}: {full_url}")
        try:
            html = fetch_content(full_url)
            soup = parse_html(html)
            yield full_url, soup
        except Exception as e:
            LOG.error(f"An error occurred while fetching words from page {n_pages}", exc_info=e)
            next_url = None
        else:
            next_url = get_next_url(soup)

    LOG.info(f"Finished iterating {n_pages} words pages")


def fetch_all_words(soup: BeautifulSoup) -> Iterator[tuple[str, str]]:
    words_list_element = soup.find('ul', {'class': "mw-allpages-chunk"})
    for word_element in words_list_element.find_all('li'):
        try:
            link_tag = next(word_element.children)
            word_str = link_tag.text
            word_link = link_tag.get('href')
            if word_str and word_link:
                yield word_str, word_link
        except Exception as e:
            LOG.error(f"An error occurred while parsing word element {word_element}", exc_info=e)


if __name__ == '__main__':
    with open('hebrew-wiktionary-words.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'url'])

        pages = iterate_words_pages()
        for page_url, page in pages:
            words = list(fetch_all_words(page))
            LOG.info(f'Found {len(words)} words in {page_url}')
            writer.writerows(words)
