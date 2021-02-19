import argparse
import csv
import json
import random
from pathlib import Path
from tqdm import tqdm

from backend.stream.common.file_util import get_file_paths


def get_argparser():
    parser = argparse.ArgumentParser(description='Jsonl minimizer')
    parser.add_argument('--input', required=True, type=lambda p: Path(p), help='input file path')
    parser.add_argument('--neg', required=True,
                        type=lambda p: Path(p), help='dir path for downloaded tweets used as negative samples')
    parser.add_argument('--neg_scale', default=9, type=int, help='number of negative samples per a positive sample')
    parser.add_argument('--output', required=True, type=lambda p: Path(p), help='output file path')
    return parser


def read_csv_file(file_path):
    data_dict = dict()
    with open(file_path, 'r', newline='') as fp:
        first = True
        for source_id, tweet_id, tweet_type, source_title, source_type, tweet, source_veracity in csv.reader(fp):
            if first:
                first = False
                continue

            if source_title not in data_dict:
                data_dict[source_title] = {'positive_ctxs': list()}
            data_dict[source_title]['positive_ctxs'].append({'title': None, 'text': tweet})
    return data_dict


def write_jsonl_file(entries, output_file_path):
    mode = 'w'
    Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file_path, mode) as fp:
        json.dump(entries, fp)


def create_dataset(data_dict, jsonl_file_paths, neg_scale):
    dict_list = list()
    for query, entry_dict in tqdm(data_dict.items()):
        entry_dict['question'] = query
        entry_dict['answers'] = list()
        entry_dict['hard_negative_ctxs'] = list()
        pos_size = len(entry_dict['positive_ctxs'])
        neg_size = int(pos_size * neg_scale)
        negative_ctx_list = list()
        while len(negative_ctx_list) < neg_size:
            random.shuffle(jsonl_file_paths)
            done = False
            for jsonl_file_path in jsonl_file_paths:
                with open(jsonl_file_path, 'r') as fp:
                    for line in fp:
                        tweet_data = json.loads(line)
                        full_text = tweet_data['full_text']
                        if len(full_text) > 0:
                            negative_ctx_list.append({'title': None, 'text': full_text})
                        if len(negative_ctx_list) >= neg_size:
                            done = True
                            break
                if done:
                    break
            if done:
                break
        entry_dict['negative_ctxs'] = negative_ctx_list
        dict_list.append(entry_dict)
    return dict_list


def main(args):
    data_dict = read_csv_file(args.input)
    jsonl_file_paths = get_file_paths(args.neg)
    dataset = create_dataset(data_dict, jsonl_file_paths, args.neg_scale)
    write_jsonl_file(dataset, args.output)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
