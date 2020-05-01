from sqlalchemy import create_engine, MetaData, Table, Column, String
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from backend.stream.utils import file_util


def create_table(table_name, engine):
    meta = MetaData()
    table = Table(
       table_name, meta,
       Column('url', String, primary_key=True),
       Column('title', String),
       Column('publishedAt', String),
       Column('addedAt', String)
    )
    meta.create_all(engine)
    return table


def update_article_url_db(article_dicts, publisher, db_file_path):
    file_util.make_parent_dirs(db_file_path)
    engine = create_engine('sqlite:///{}'.format(db_file_path), echo=True)
    base_cls = declarative_base()

    class Article(base_cls):
        __tablename__ = publisher

        url = Column(String, primary_key=True)
        title = Column(String)
        publishedAt = Column(String)
        addedAt = Column(String)

    # Create table
    base_cls.metadata.create_all(bind=engine)

    # Add articles to the table
    articles = [Article(**article_dict) for article_dict in article_dicts]
    session = sessionmaker(bind=engine)()
    try:
        session.add_all(articles)
        session.commit()
    except SQLAlchemyError as e:
        print(e)
    finally:
        session.close()
