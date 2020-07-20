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


def modify_records(custom_type, records, table_class_name, record_ids=None):
    record_list = list()
    if custom_type == 'initial_labeled_tweet':
        # Assuming input/output tables are empty, and model/label/misinformation tables are filled
        for _, misconception_id, misconception, tweet, bert_score, label, tweet_id in records:
            if len(label) == 0 or len(tweet_id) == 0:
                continue

            record = dict()
            if table_class_name == 'Input':
                record = {'id': len(record_list) + 1, 'source_type': 'Twitter', 'source_id': tweet_id, 'text': tweet}
            elif table_class_name == 'Output':
                record = {'input_id': len(record_list) + 1, 'confidence': 1, 'model_id': 'Arjuna',
                          'misinfo_id': misconception_id,
                          'label_id': 0 if label == 'pos' else 1 if label == 'neg' else 2}
            if len(record) > 0:
                record_list.append(record)
    elif custom_type == 'old_csv_format':
        # header won't be used
        if record_ids is None:
            records.pop(0)
        else:
            assert len(records) == len(record_ids), \
                '# records {} should be equal to # record_ids {}'.format(len(records), len(record_ids))

        for i, (misconception_id, misconception, tweet, label, tweet_id) in enumerate(records):
            if len(label) == 0 or len(tweet_id) == 0:
                continue

            record = dict()
            if table_class_name == 'Input':
                record = {'source_type': 'Twitter', 'source_id': tweet_id, 'text': tweet}
            elif table_class_name == 'Output':
                record = {'input_id': record_ids[i], 'confidence': 1, 'model_id': 'Arjuna',
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
        record_ids = None
        for table_class_name in table_class_names:
            record_dicts = modify_records(args.custom, records, table_class_name, record_ids)
            record_ids = insert_records(record_dicts, table_class_name, args.db)



if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
