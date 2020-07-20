import argparse
import json
from pathlib import Path

from backend.stream.db.operation import add_records
from backend.stream.db.util import get_engine


def get_argparser():
    parser = argparse.ArgumentParser(description='JSONL to DB')
    parser.add_argument('--input', required=True, type=lambda p: Path(p), help='input JSONL file path')
    parser.add_argument('--table', required=True, help='table class name')
    parser.add_argument('--custom', help='modify records before adding, for a specific type of input file')
    parser.add_argument('--db', required=True, type=lambda p: Path(p), help='output DB file path')
    parser.add_argument('--output', default=Path('./record_ids.txt'), type=lambda p: Path(p),
                        help='output file path with inserted IDs in the target table')
    return parser


def modify_records(custom_type, records):
    record_list = list()
    if custom_type == 'initial_wiki':
        for i in range(len(records)):
            old_record = records[i]
            record = {'id': old_record.pop('id'), 'text': old_record.pop('canonical_sentence'),
                      'source': old_record.pop('origin'), 'reliability': 1,
                      'url': json.dumps({'list': old_record.pop('sources')})}
            record['misc'] = json.dumps(old_record)
            record_list.append(record)
    elif custom_type == 'old_jsonl_format':
        for i in range(len(records)):
            old_record = records[i]
            if 'id' in old_record:
                old_record.pop('id')

            record = {'text': old_record.pop('canonical_sentence'),
                      'source': old_record.pop('origin'), 'reliability': 1,
                      'url': json.dumps({'list': old_record.pop('sources')})}
            record['misc'] = json.dumps(old_record)
            record_list.append(record)
    return record_list


def insert_records(records, table_class_name, db_file_path, output_file_path):
    engine = get_engine(db_file_path)
    _, record_ids = add_records(records, engine, table_class_name, returns_id=True)
    Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file_path, 'w') as fp:
        for record_id in record_ids:
            fp.write('{}\n'.format(record_id))


def main(args):
    with open(args.input, 'r') as fp:
        records = [json.loads(line) for line in fp]

    if args.custom is not None:
        records = modify_records(args.custom, records)
    insert_records(records, args.table, args.db, args.output)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
