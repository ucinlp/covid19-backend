import argparse
import json
from pathlib import Path

REF_DICT = {
    'id': None, 'text': None, 'favorite_count': None, 'retweet_count': None, 'lang': None,
    'user': {'followers_count': None, 'friends_count': None}, 'created_at': None, 'coordinate': None
}


def get_argparser():
    parser = argparse.ArgumentParser(description='Jsonl minimizer for Chen\'s Tweet dataset')
    parser.add_argument('--input', required=True, type=lambda p: Path(p), help='input file path')
    parser.add_argument('--output', required=True, type=lambda p: Path(p), help='output file path')
    return parser


def build_sub_dict(main_dict, ref_dict):
    sub_dict = dict()
    for key, value in ref_dict.items():
        sub_dict[key] = build_sub_dict(main_dict[key], value) if isinstance(value, dict) \
            else main_dict.get(key, value)
    return sub_dict


def write_jsonl_file(json_objs, output_file_path, first=True):
    mode = 'w' if first else 'a+'
    Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file_path, mode) as fp:
        for json_obj in json_objs:
            fp.write('{}\n'.format(json.dumps(json_obj)))


def process_single_jsonl(input_file_path, output_file_path, buffer_size=10000):
    buffer_list = list()
    first = True
    with open(input_file_path, 'r') as fp:
        for line in fp:
            main_dict = json.loads(line)
            sub_dict = build_sub_dict(main_dict, REF_DICT)
            buffer_list.append(sub_dict)
            if len(buffer_list) >= buffer_size:
                write_jsonl_file(buffer_list, output_file_path, first=first)
                buffer_list.clear()
                first = False
        write_jsonl_file(buffer_list, output_file_path, first=first)


def main(args):
    process_single_jsonl(args.input, args.output)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
