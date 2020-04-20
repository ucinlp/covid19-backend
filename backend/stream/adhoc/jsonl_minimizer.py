import argparse
import json
import os

import yaml

from backend.stream.utils.file_util import get_dir_paths, get_file_paths, make_parent_dirs


def get_argparser():
    parser = argparse.ArgumentParser(description='Jsonl minimizer')
    parser.add_argument('--input', required=True, help='input dir/file path')
    parser.add_argument('--config', required=True, help='config file path')
    parser.add_argument('--output', required=True, help='output dir/file path')
    return parser


def build_sub_dict(main_dict, ref_dict):
    sub_dict = dict()
    for key, value in ref_dict.items():
        if isinstance(value, dict):
            sub_dict[key] = build_sub_dict(main_dict[key], value)
        elif value is True and key in main_dict:
            sub_dict[key] = main_dict[key]
    return sub_dict


def write_jsonl_file(json_objs, output_file_path, first=True):
    mode = 'w' if first else 'a+'
    make_parent_dirs(output_file_path)
    with open(output_file_path, mode) as fp:
        for json_obj in json_objs:
            fp.write('{}\n'.format(json.dumps(json_obj)))


def process_single_jsonl(input_file_path, config, output_file_path, buffer_size=10000):
    buffer_list = list()
    first = True
    with open(input_file_path, 'r') as fp:
        for line in fp:
            main_dict = json.loads(line)
            sub_dict = build_sub_dict(main_dict, config)
            buffer_list.append(sub_dict)
            if len(buffer_list) >= buffer_size:
                write_jsonl_file(buffer_list, output_file_path, first=first)
                buffer_list.clear()
                first = False
        write_jsonl_file(buffer_list, output_file_path, first=first)


def process_multiple_jsonls(input_dir_path, config, output_dir_path):
    for input_file_path in get_file_paths(input_dir_path):
        output_file_path = os.path.join(output_dir_path, os.path.basename(input_file_path))
        process_single_jsonl(input_file_path, config, output_file_path)


def main(args):
    with open(args.config, 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    is_file = os.path.isfile(args.input)
    if is_file:
        process_single_jsonl(args.input, config, args.output)
    else:
        input_dir_paths = get_dir_paths(args.input)
        output_root_dir_path = args.output
        for input_dir_path in input_dir_paths:
            output_dir_path = os.path.join(output_root_dir_path, os.path.basename(input_dir_path))
            process_multiple_jsonls(input_dir_path, config, output_dir_path)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
