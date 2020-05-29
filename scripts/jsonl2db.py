import argparse
import json
from pathlib import Path

from backend.stream.db.operation import add_entities
from backend.stream.db.util import get_engine


def get_argparser():
    parser = argparse.ArgumentParser(description='JSONL to DB')
    parser.add_argument('--input', required=True, type=lambda p: Path(p), help='input JSONL file path')
    parser.add_argument('--table', required=True, help='table class name')
    parser.add_argument('--db', required=True, type=lambda p: Path(p), help='output DB file path')
    return parser


def add_records(records, table_class_name, db_file_path):
    engine = get_engine(db_file_path)
    add_entities(records, engine, table_class_name)


def main(args):
    with open(args.input, 'r') as fp:
        records = [json.loads(line) for line in fp]
    add_records(records, args.table, args.db)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
