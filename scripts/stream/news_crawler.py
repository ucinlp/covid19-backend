import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from sqlalchemy import Column, DateTime, String
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from backend.stream.apis.diffbot import DiffbotArticleClient
from backend.stream.apis.news_api import NewsApiClient
from backend.stream.common.misc_util import overwrite_config
from backend.stream.db.util import get_engine


def get_argparser():
    parser = argparse.ArgumentParser(description='News crawler')
    parser.add_argument('--config', required=True, type=lambda p: Path(p), help='config file path')
    parser.add_argument('--json', help='dictionary to overwrite config')
    parser.add_argument('--tol', type=int, default=0, help='maximum number of News API errors you can tolerate')
    parser.add_argument('--db', required=True, type=lambda p: Path(p), help='output DB file path')
    parser.add_argument('--output', required=True, type=lambda p: Path(p), help='output root dir path')
    return parser


def update_article_url_db(article_dicts, publisher, engine):
    base_cls = declarative_base()

    class Article(base_cls):
        __tablename__ = publisher

        url = Column(String, primary_key=True)
        title = Column(String, nullable=False)
        publishedAt = Column(String, nullable=False)
        addedAt = Column(DateTime, nullable=False, default=datetime.utcnow)

    if not engine.has_table(publisher):
        base_cls.metadata.create_all(bind=engine)

    # Add articles to the table
    session = sessionmaker(bind=engine)()
    article_list = list()
    article_url_list = list()
    done_url_set = set()
    for article_dict in article_dicts:
        article_url = article_dict['url']
        if article_url not in done_url_set \
                and session.query(Article.url).filter_by(url=article_url).scalar() is None:
            article_list.append(Article(**article_dict))
            article_url_list.append(article_url)
            done_url_set.add(article_url)
    try:
        session.add_all(article_list)
        session.commit()
    except SQLAlchemyError as e:
        print(e)
    finally:
        session.close()
    return article_url_list


def get_related_article_urls(news_api_client, news_api_config, max_tol, category, db_file_path):
    article_dict_list = list()
    endpoint = news_api_config['endpoint']
    params_config = news_api_config['params']
    num_hits = -1
    article_count = 0
    page_count = 1
    failure_count = 0
    while num_hits == -1 or article_count < num_hits:
        try:
            result = news_api_client.fetch(endpoint, page=page_count, **params_config)
            if result['status'] == 'error' and result['code'] == 'maximumResultsReached':
                break

            num_hits = result['totalResults']
            articles = result['articles']
            article_count += len(articles)
            for article in articles:
                article_dict_list.append({'url': article['url'], 'title': article['title'],
                                          'publishedAt': article.get('publishedAt', '')})
        except Exception:
            failure_count += 1
            if failure_count > max_tol:
                break
            news_api_client = NewsApiClient()
        page_count += 1

    engine = get_engine(db_file_path, echo=False)
    article_urls = update_article_url_db(article_dict_list, category, engine)
    return article_urls


def download_article_bodies(diffbot_client, article_urls, diffbot_config):
    article_body_list = list()
    params_config = diffbot_config['params']
    for article_url in article_urls:
        try:
            article_with_body = diffbot_client.fetch(article_url, **params_config)
            # As of Apr 17, 2020, "At the moment, only a single object will be returned for Article API requests."
            if 'errorCode' not in article_with_body:
                article_body_list.append(article_with_body)
        except Exception:
            print('Diffbot error: {}'.format(sys.exc_info()[0]))
    return article_body_list


def write_jsonl_file(articles, output_file_path):
    Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file_path, 'w') as fp:
        for article in articles:
            fp.write('{}\n'.format(json.dumps(article)))
    print('{} new articles were stored at {}'.format(len(articles), output_file_path))


def main(args):
    with open(args.config, 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    if args.json is not None:
        overwrite_config(config, args.json)

    category = config['category']
    timestamp = datetime.utcnow().strftime('utc-%Y%m%d-%H%M%S')
    news_api_client = NewsApiClient()
    article_urls = get_related_article_urls(news_api_client, config['news_api'],
                                            args.tol, category, os.path.abspath(args.db))
    diffbot_client = DiffbotArticleClient()
    articles = download_article_bodies(diffbot_client, article_urls, config['diffbot'])
    write_jsonl_file(articles, os.path.join(args.output, category, timestamp + '.jsonl'))


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
