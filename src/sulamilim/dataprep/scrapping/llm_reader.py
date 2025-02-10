from main_content_extractor import MainContentExtractor

from sulamilim.dataprep.scrapping.fetch import fetch_content


def parse_for_llm(html: str) -> str:
    return MainContentExtractor.extract(html, output_format='markdown', include_links=False)



if __name__ == '__main__':
    url = "https://he.wiktionary.org/wiki/%D7%94%D7%9C%D7%99%D7%9E%D7%94#%D7%9E%D7%99%D7%9C%D7%99%D7%9D_%D7%A0%D7%A8%D7%93%D7%A4%D7%95%D7%AA"
    content = fetch_content(url)
    md = parse_for_llm(content)
    print(md)