import argparse
import gzip
import json
import os
from datetime import date, timedelta
from pathlib import Path

import yaml
from requests.exceptions import Timeout
from tweepy import OAuthHandler, API, Cursor

CONSUMER_KEY = os.environ.get('HIV_TWITTER_CONSUMER_KEY', None)
CONSUMER_SECRET = os.environ.get('HIV_TWITTER_CONSUMER_SECRET', None)
ACCESS_TOKEN = os.environ.get('HIV_TWITTER_ACCESS_TOKEN', None)
ACCESS_TOKEN_SECRET = os.environ.get('HIV_TWITTER_ACCESS_TOKEN_SECRET', None)


def get_argparser():
    parser = argparse.ArgumentParser(description='COVID-19 Tweet dataset downloader')
    parser.add_argument('--config', required=True, type=lambda p: Path(p), help='config file path')
    parser.add_argument('--since', type=str, help='tweets since YYYY-MM-DD')
    parser.add_argument('--until', type=str, help='tweets until YYYY-MM-DD')
    parser.add_argument('--output', required=True, type=lambda p: Path(p), help='output root dir path')
    return parser


def get_tweepy_api():
    auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    tweepy_api = API(auth, wait_on_rate_limit=True)
    return tweepy_api


def write_gzip_jsonl_file(tweets, output_file_path, first=True):
    Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
    write_mode = 'wt' if first else 'at'
    if len(tweets) > 0:
        with gzip.open(output_file_path, write_mode) as fp:
            for json_obj in tweets:
                fp.write('{}\n'.format(json.dumps(json_obj)))


def crawl_tweets(tweepy_api, query_config, save_config, since_date, until_date, output_dir_path):
    output_dir_path = os.path.expanduser(output_dir_path)
    output_prefix = save_config['output_prefix']
    batch_size = save_config['batch_size']
    batch_index = 0
    tweet_count = 0
    today_date = date.today()
    if since_date is None and until_date is None:
        since_date = str(today_date - timedelta(days=1))
        until_date = str(today_date)
    elif since_date is None:
        since_date = str(date.fromisoformat(until_date) - timedelta(days=1))
    elif until_date is None:
        until_date = str(date.fromisoformat(since_date) + timedelta(days=1))

    try:
        search_cursor = Cursor(tweepy_api.search, since=since_date, until=until_date, **query_config).items()
        tweet_list = list()
        for tweet in search_cursor:
            tweet_list.append(tweet._json)
            if len(tweet_list) == batch_size:
                output_file_path =\
                    os.path.join(output_dir_path, '{}-{}-{}.jsonl.gz'.format(output_prefix, since_date, batch_index))
                write_gzip_jsonl_file(tweet_list, output_file_path)
                batch_index += 1
                tweet_count += len(tweet_list)
                tweet_list.clear()

        if len(tweet_list) > 0:
            output_file_path =\
                os.path.join(output_dir_path, '{}-{}-{}.jsonl.gz'.format(output_prefix, since_date, batch_index))
            write_gzip_jsonl_file(tweet_list, output_file_path)
            batch_index += 1
            tweet_count += len(tweet_list)
            tweet_list.clear()
    except Timeout:
        print('Timeout error')
    print('{} tweets were downloaded'.format(tweet_count))


def main(args):
    with open(args.config, 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    tweepy_api = get_tweepy_api()
    crawl_tweets(tweepy_api, config['query'], config['save'], args.since, args.until, args.output)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
