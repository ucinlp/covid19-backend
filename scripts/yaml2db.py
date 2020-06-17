import argparse
from pathlib import Path

import yaml

from backend.stream.common.misc_util import overwrite_config
from backend.stream.db.operation import add_records
from backend.stream.db.util import get_engine


def get_argparser():
    parser = argparse.ArgumentParser(description='YAML to DB')
    parser.add_argument('--config', required=True, type=lambda p: Path(p), help='config file path')
    parser.add_argument('--json', help='dictionary to overwrite config')
    parser.add_argument('--db', required=True, type=lambda p: Path(p), help='output DB file path')
    return parser


def insert_records(records_dict, table_class_name, db_file_path):
    engine = get_engine(db_file_path)
    add_records(records_dict.values(), engine, table_class_name)


def main(args):
    with open(args.config, 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    if args.json is not None:
        overwrite_config(config, args.json)
    insert_records(config['records'], config['table_class'], args.db)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
