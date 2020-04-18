import argparse
import json

import yaml

from backend.stream.apis.diffbot import DiffbotArticleClient
from backend.stream.apis.news_api import NewsApiClient
from backend.stream.utils.file_util import make_parent_dirs
from backend.stream.utils.misc_util import overwrite_config


def get_argparser():
    parser = argparse.ArgumentParser(description='News crawler')
    parser.add_argument('--config', required=True, help='config file path')
    parser.add_argument('--json', help='dictionary to overwrite config')
    parser.add_argument('--output', required=True, help='output file path')
    return parser


def get_related_article_urls(news_api_config):
    article_url_list = list()
    news_api_client = NewsApiClient()
    endpoint = news_api_config['endpoint']
    params_config = news_api_config['params']
    num_hits = -1
    article_count = 0
    page_count = 1
    failure_count = 0
    while num_hits == -1 or article_count < num_hits:
        try:
            result = news_api_client.fetch(endpoint, page=page_count, **params_config)
            num_hits = result['totalResults']
            articles = result['articles']
            article_count += len(articles)
            for article in articles:
                article_url_list.append(article['url'])
        except:
            failure_count += 1
            if failure_count > 5:
                break
            news_api_client = NewsApiClient()
        page_count += 1
    return article_url_list


def download_article_bodies(article_urls, diffbot_config):
    article_body_list = list()
    diffbot_client = DiffbotArticleClient()
    params_config = diffbot_config['params']
    for article_url in article_urls:
        try:
            article_with_body = diffbot_client.fetch(article_url, **params_config)
            # As of Apr 17, 2020, "At the moment, only a single object will be returned for Article API requests."
            article_body_list.append(article_with_body)
        except Exception as e:
            print(e)
    return article_body_list


def main(args):
    with open(args.config, 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    if args.json is not None:
        overwrite_config(config, args.json)

    article_urls = get_related_article_urls(config['news_api'])
    articles = download_article_bodies(article_urls, config['diffbot'])
    output_file_path = args.output
    make_parent_dirs(output_file_path)
    with open(output_file_path, 'w') as fp:
        for article in articles:
            fp.write('{}\n'.format(json.dumps(article)))


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
