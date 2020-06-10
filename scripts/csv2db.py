import argparse
import csv
from pathlib import Path

from backend.stream.db.operation import add_entities
from backend.stream.db.util import get_engine


def get_argparser():
    parser = argparse.ArgumentParser(description='CSV to DB')
    parser.add_argument('--input', required=True, type=lambda p: Path(p), help='input csv file path')
    parser.add_argument('--table', required=True, help='table class name')
    parser.add_argument('--custom', required=True,
                        help='modify records before adding, for a specific type of input file')
    parser.add_argument('--db', required=True, type=lambda p: Path(p), help='output DB file path')
    return parser


def modify_records(custom_type, records, table_class_name):
    record_list = list()
    if custom_type == 'initial_labeled_tweet':
        # Assuming input/output tables are empty, and model/label/misinformation tables are filled
        for _, misconception_id, misconception, tweet, bert_score, label, tweet_id in range(len(records)):
            if len(label) == 0 or len(tweet_id) == 0:
                continue

            record = {'source': 'Twitter', 'source_id': tweet_id, 'text': tweet}
            if table_class_name == 'Output':
                record['confidence'] = 1
                record['Arjuna'] = 1
                record['misinfo_id'] = misconception_id
                record['label_id'] = 0 if label == 'pos' else 1 if label == 'neg' else 2
            elif table_class_name != 'Input':
                continue
            record_list.append(record)
    return record_list


def add_records(records, table_class_name, db_file_path):
    engine = get_engine(db_file_path)
    add_entities(records, engine, table_class_name)


def main(args):
    with open(args.input, 'r') as fp:
        records = [row for row in csv.reader(fp)]

    table_class_name = args.table
    if args.custom is not None:
        records = modify_records(args.custom, records, table_class_name)
    add_records(records, table_class_name, args.db)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())