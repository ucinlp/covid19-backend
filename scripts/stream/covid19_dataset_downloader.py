import argparse
import json
import os
import sys
import time
from pathlib import Path

from requests.exceptions import Timeout
from requests_oauthlib import OAuth1Session

from backend.stream.common.file_util import get_dir_paths, get_file_paths

CONSUMER_KEY = os.environ.get('TWITTER_CONSUMER_KEY', None)
CONSUMER_SECRET = os.environ.get('TWITTER_CONSUMER_SECRET', None)
ACCESS_TOKEN = os.environ.get('TWITTER_ACCESS_TOKEN', None)
ACCESS_TOKEN_SECRET = os.environ.get('TWITTER_ACCESS_TOKEN_SECRET', None)


def get_argparser():
    parser = argparse.ArgumentParser(description='COVID-19 Tweet dataset downloader')
    parser.add_argument('--input', required=True, type=lambda p: Path(p), help='input root dir path')
    parser.add_argument('--batch_size', default=100, type=int, help='number of tweets per request')
    parser.add_argument('--output', required=True, type=lambda p: Path(p), help='output root dir path')
    return parser


def get_done_set(output_dir_path):
    file_paths = get_file_paths(output_dir_path, ext='.jsonl')
    done_set = set()
    for file_path in file_paths:
        if os.stat(file_path).st_size > 0:
            done_set.add(file_path)
    return done_set


def send_query(client, ids_str):
    url = 'https://api.twitter.com/1.1/statuses/lookup.json'
    params = {'tweet_mode': 'extended', 'id': ids_str}
    try:
        req = client.get(url, params=params)
        if req.status_code == 200:
            return json.loads(req.text)
        return req.status_code
    except Timeout:
        print('Timeout error')
        return 504
    except Exception:
        print('Twitter API error: {}'.format(sys.exc_info()[0]))
    return None


def download_tweet_data(input_dir_path, batch_size, output_dir_path):
    if not os.path.isdir(output_dir_path):
        os.makedirs(output_dir_path)
        print('Created {}'.format(output_dir_path))

    done_set = get_done_set(output_dir_path)
    client = OAuth1Session(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    for input_file_path in get_file_paths(input_dir_path, ext='.txt'):
        output_file_path =\
            os.path.join(output_dir_path, os.path.basename(input_file_path).replace('.txt', '.jsonl'))
        if output_file_path in done_set:
            print('`{}` already exists. Download process is skipped.'.format(output_file_path))
            continue

        with open(input_file_path, 'r') as fp:
            tweet_ids = [line.strip() for line in fp]

        index = 0
        request_count = 0
        json_list = list()
        print('Processing {}'.format(input_file_path))
        while index < len(tweet_ids):
            ids_str = ','.join(tweet_ids[index: index + batch_size])
            tweet_data = send_query(client, ids_str)
            request_count += 1
            if isinstance(tweet_data, list):
                json_list.extend(tweet_data)
                index += batch_size
            elif tweet_data == 429:
                print('{} requests were sent after the interval'.format(request_count))
                print('Sleeping for 15 min')
                # With standard APIs, 30 requests / min
                time.sleep(60.0 * 15)
                request_count = 0
            elif tweet_data == 504:
                client = OAuth1Session(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
            else:
                print('Error code: {}'.format(tweet_data))

        with open(output_file_path, 'w') as fp:
            for json_obj in json_list:
                fp.write('{}\n'.format(json.dumps(json_obj)))


def main(args):
    input_dir_paths = get_dir_paths(args.input)
    batch_size = args.batch_size
    output_root_dir_path = args.output
    for input_dir_path in input_dir_paths:
        output_dir_path = os.path.join(output_root_dir_path, os.path.basename(input_dir_path))
        download_tweet_data(input_dir_path, batch_size, output_dir_path)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
