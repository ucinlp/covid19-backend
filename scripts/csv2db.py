import argparse
import csv
from pathlib import Path

from backend.stream.db.operation import add_records
from backend.stream.db.util import get_engine


def get_argparser():
    parser = argparse.ArgumentParser(description='CSV to DB')
    parser.add_argument('--input', required=True, type=lambda p: Path(p), help='input csv file path')
    parser.add_argument('--tables', required=True, metavar='N', nargs='+', help='list of table class names')
    parser.add_argument('--custom', help='modify records before adding, for a specific type of input file')
    parser.add_argument('--db', required=True, type=lambda p: Path(p), help='output DB file path')
    return parser


def modify_records(custom_type, records, table_class_name, srcid2inputid_dict=None):
    record_list = list()
    tweet_id_set = set()
    if custom_type == 'old_csv_format':
        for i, (_, misconception_id, misconception, tweet, _, label, tweet_id) in enumerate(records):
            if len(label) == 0 or len(tweet_id) == 0 or tweet_id in tweet_id_set:
                continue

            record = dict()
            if table_class_name == 'Input':
                record = {'source_type': 'Twitter', 'source_id': tweet_id, 'text': tweet}
                tweet_id_set.add(tweet_id)
            elif table_class_name == 'Output':
                record = {'input_id': srcid2inputid_dict[tweet_id], 'confidence': 1, 'model_id': 'Arjuna',
                          'misinfo_id': misconception_id,
                          'label_id': 0 if label == 'pos' else 1 if label == 'neg' else 2}
            if len(record) > 0:
                record_list.append(record)
    return record_list


def insert_records(records, table_class_name, db_file_path):
    engine = get_engine(db_file_path)
    _, record_ids = add_records(records, engine, table_class_name, returns_id=True)
    return record_ids


def main(args):
    with open(args.input, 'r', encoding='utf8') as fp:
        records = [row for row in csv.reader(fp)]

    table_class_names = args.tables
    if len(table_class_names) == 1:
        table_class_name = table_class_names[0]
        records = modify_records(args.custom, records, table_class_name)
        insert_records(records, table_class_name, args.db)
    elif len(table_class_names) == 2:
        srcid2inputid_dict = dict()
        for table_class_name in table_class_names:
            record_dicts = modify_records(args.custom, records, table_class_name, srcid2inputid_dict)
            record_ids = insert_records(record_dicts, table_class_name, args.db)
            if len(srcid2inputid_dict) == 0:
                for record_id, record_dict in zip(record_ids, record_dicts):
                    srcid2inputid_dict[record_dict['source_id']] = record_id


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
