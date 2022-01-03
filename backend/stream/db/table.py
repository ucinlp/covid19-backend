from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

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
    human = Column(Boolean, nullable=False)
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
    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(String, nullable=False)
    url = Column(JSON, nullable=True)
    source = Column(String, nullable=False)
    reliability = Column(Float, nullable=False)
    misc = Column(JSON, nullable=True)
    date = Column(DateTime, nullable=False, default=datetime.utcnow)

    @classmethod
    def check_if_exists(cls, entity, session):
        return session.query(cls.text).filter_by(text=entity.text).scalar() is not None


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
        # Should be used before inserting new records
        return session.query(cls.source_type).filter_by(source_id=entity.source_id).scalar() is not None


@register_table_class
class Output(base_cls):
    __tablename__ = 'output'
    id = Column(Integer, primary_key=True, autoincrement=True)
    input_id = Column(Integer, ForeignKey('input.id'), nullable=False)
    model_id = Column(String, ForeignKey('model.id'), nullable=False)
    misinfo_id = Column(Integer, ForeignKey('misinformation.id'), nullable=False)
    label_id = Column(Integer, ForeignKey('label.id'), nullable=False)
    confidence = Column(Float, nullable=False)
    misc = Column(JSON, nullable=True)
    date = Column(DateTime, nullable=False, default=datetime.utcnow)

    input = relationship('Input', single_parent=True)
    model = relationship('Model')
    misinformation = relationship('Misinformation')
    label = relationship('Label')

    @classmethod
    def check_if_exists(cls, entity, session):
        return session.query(cls.id).filter_by(id=entity.id).scalar() is not None


def get_table_class(cls_name):
    if cls_name not in TABLE_CLASS_DICT:
        raise KeyError('cls_name `{}` is not expected'.format(cls_name))
    return TABLE_CLASS_DICT[cls_name]

