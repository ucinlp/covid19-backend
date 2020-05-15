from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, Column, DateTime, Float, JSON, String
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


def update_article_url_db(article_dicts, publisher, db_file_path):
    Path(db_file_path).parent.mkdir(parents=True, exist_ok=True)
    base_cls = declarative_base()

    class Article(base_cls):
        __tablename__ = publisher

        url = Column(String, primary_key=True)
        title = Column(String, nullable=False)
        publishedAt = Column(String, nullable=False)
        addedAt = Column(DateTime, nullable=False, default=datetime.utcnow)

    engine = create_engine('sqlite:///{}'.format(db_file_path), echo=False)
    base_cls.metadata.create_all(bind=engine)

    # Add articles to the table
    session = sessionmaker(bind=engine)()
    article_list = list()
    article_url_list = list()
    for article_dict in article_dicts:
        article_url = article_dict['url']
        if session.query(Article.url).filter_by(url=article_url).scalar() is None:
            article_list.append(Article(**article_dict))
            article_url_list.append(article_url)
    try:
        session.add_all(article_list)
        session.commit()
    except SQLAlchemyError as e:
        print(e)
    finally:
        session.close()
    return article_url_list


def update_misinfo_db(entities, model_id, db_file_path):
    Path(db_file_path).parent.mkdir(parents=True, exist_ok=True)
    base_cls = declarative_base()

    class Misinfo(base_cls):
        __tablename__ = model_id

        id = Column(String, primary_key=True)
        confidence = Column(Float)
        date = Column(DateTime, default=datetime.utcnow)
        misc = Column(JSON)

    engine = create_engine('sqlite:///{}'.format(db_file_path), echo=True)
    base_cls.metadata.create_all(bind=engine)

    # Add misinformation to the table
    session = sessionmaker(bind=engine)()
    added_entity_list = list()
    for entity in entities:
        entity_id = entity['id']
        if session.query(Misinfo.id).filter_by(id=entity_id).scalar() is None:
            added_entity_list.append(Misinfo(**entity))
    try:
        session.add_all(added_entity_list)
        session.commit()
    except SQLAlchemyError as e:
        print(e)
    finally:
        session.close()
    return added_entity_list
