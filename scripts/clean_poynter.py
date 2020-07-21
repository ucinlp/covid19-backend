import json
from pathlib import Path

from bs4 import BeautifulSoup


ROOT_DIR = Path('poynter_html')


def main():
    for file in ROOT_DIR.iterdir():
        with open(file, 'r') as f:
            html = f.read()
        soup = BeautifulSoup(html)
        articles = soup.find_all('article')
        for article in articles:
            id_ = article['id']
            card_title = article.header.h2.a
            source = card_title['href']
            status, claim = tuple(card_title.strings)
            claim = ' '.join(claim.split())

            obj = {
                'id': id_,
                'canonical_sentence': claim,
                'origin': 'Poynter',
                'reliability_score': 'na',
                'category': [status[:-1]],
                'sources': [source],
                'pos_variations': [claim],
                'neg_variations': []
            }
            print(json.dumps(obj))


if __name__ == '__main__':
    main()
