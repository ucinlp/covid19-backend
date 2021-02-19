from pathlib import Path

from sqlalchemy import create_engine


def get_engine(db_file_path, echo=False):
    Path(db_file_path).parent.mkdir(parents=True, exist_ok=True)
    return create_engine('sqlite:///{}'.format(db_file_path), echo=echo)

