from datetime import datetime

from sqlalchemy import Column, Integer, DateTime, Float, JSON, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

base_cls = declarative_base()
TABLE_CLASS_DICT = dict()


def register_table_class(cls):
    TABLE_CLASS_DICT[cls.__name__] = cls
    return cls


@register_table_class
class Label(base_cls):
    __tablename__ = 'label'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    misc = Column(JSON, nullable=True)
    date = Column(DateTime, nullable=False, default=datetime.utcnow)

    @classmethod
    def check_if_exists(cls, entity, session):
        return session.query(cls.id).filter_by(id=entity.id).scalar() is not None


@register_table_class
class Reference(base_cls):
    __tablename__ = 'reference'
    url = Column(String, primary_key=True)
    misc = Column(JSON, nullable=True)
    date = Column(DateTime, nullable=False, default=datetime.utcnow)

    @classmethod
    def check_if_exists(cls, entity, session):
        return session.query(cls.url).filter_by(url=entity.url).scalar() is not None


@register_table_class
class Model(base_cls):
    __tablename__ = 'model'
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    config = Column(JSON, nullable=False)
    desc = Column(String, nullable=True)
    date = Column(DateTime, nullable=False, default=datetime.utcnow)

    @classmethod
    def check_if_exists(cls, entity, session):
        return session.query(cls.id).filter_by(id=entity.id).scalar() is not None


@register_table_class
class Source(base_cls):
    __tablename__ = 'source'
    type = Column(String, primary_key=True)
    date = Column(DateTime, nullable=False, default=datetime.utcnow)

    @classmethod
    def check_if_exists(cls, entity, session):
        return session.query(cls.type).filter_by(type=entity.type).scalar() is not None


@register_table_class
class Misinformation(base_cls):
    __tablename__ = 'misinformation'
    text = Column(String, primary_key=True)
    model_id = Column(String, ForeignKey('model.id'), primary_key=True)
    label_id = Column(String, ForeignKey('label.id'), nullable=False)
    date = Column(DateTime, nullable=False, default=datetime.utcnow)

    @classmethod
    def check_if_exists(cls, entity, session):
        return session.query(cls.text).filter_by(text=entity.text, model_id=entity.model_id).scalar() is not None


@register_table_class
class Input(base_cls):
    __tablename__ = 'input'
    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(String, nullable=False)
    source_type = Column(String, ForeignKey('source.type'), nullable=False)
    source_id = Column(String, nullable=False)
    misc = Column(JSON, nullable=True)
    date = Column(DateTime, nullable=False, default=datetime.utcnow)

    @classmethod
    def check_if_exists(cls, entity, session):
        return session.query(cls.id).filter_by(id=entity.id).scalar() is not None


@register_table_class
class Output(base_cls):
    __tablename__ = 'output'
    id = Column(Integer, primary_key=True, autoincrement=True)
    input_id = Column(Integer, ForeignKey('input.id'), nullable=False)
    model_id = Column(String, ForeignKey('model.id'), nullable=False)
    prediction = Column(String, ForeignKey('label.id'), nullable=False)
    confidence = Column(Float, nullable=False)
    misc = Column(JSON, nullable=True)
    date = Column(DateTime, nullable=False, default=datetime.utcnow)

    @classmethod
    def check_if_exists(cls, entity, session):
        return session.query(cls.id).filter_by(id=entity.id).scalar() is not None


def get_table_class(cls_name):
    if cls_name not in TABLE_CLASS_DICT:
        raise KeyError('cls_name `{}` is not expected'.format(cls_name))
    return TABLE_CLASS_DICT[cls_name]
