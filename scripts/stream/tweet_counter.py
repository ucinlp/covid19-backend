import argparse
import gzip
import json

from backend.stream.common.file_util import get_file_paths


def get_argparser():
    parser = argparse.ArgumentParser(description='Downloaded tweet counter')
    parser.add_argument('--input', required=True, nargs='+', help='input dir path(s)')
    return parser


def get_all_file_paths(dir_paths):
    file_path_list = list()
    for dir_path in dir_paths:
        file_path_list.extend(get_file_paths(dir_path))
    return file_path_list


def count_tweets(file_paths):
    tweet_id_set = set()
    tweet_text_set = set()
    user_id_set = set()

    for file_path in file_paths:
        open_func = None
        mode = None
        if file_path.endswith('.gz'):
            open_func = gzip.open
            mode = 'rt'
        elif file_path.endswith('.jsonl'):
            open_func = open
            mode = 'r'
        else:
            continue

        with open_func(file_path, mode) as fp:
            for line in fp:
                tweet = json.loads(line)
                tweet_id_set.add(tweet['id_str'])
                tweet_text_set.add(tweet['full_text'])
                user_id_set.add(tweet['user']['id_str'])

    print(f'# files: {len(file_paths)}')
    print(f'# unique tweet IDs: {len(tweet_id_set)}')
    print(f'# unique tweet texts: {len(tweet_text_set)}')
    print(f'# unique user IDs: {len(user_id_set)}')


def main(args):
    input_file_paths = get_all_file_paths(args.input)
    count_tweets(input_file_paths)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
