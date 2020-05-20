from datetime import datetime

from sqlalchemy import Column, Integer, DateTime, Float, JSON, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

base_cls = declarative_base()


class Label(base_cls):
    __tablename__ = 'label'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    misc = Column(JSON, nullable=True)
    date = Column(DateTime, nullable=False, default=datetime.utcnow)


class Reference(base_cls):
    __tablename__ = 'reference'
    url = Column(String, primary_key=True)
    misc = Column(JSON, nullable=True)
    date = Column(DateTime, nullable=False, default=datetime.utcnow)


class Model(base_cls):
    __tablename__ = 'model'
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    config = Column(JSON, nullable=False)
    desc = Column(String, nullable=True)
    date = Column(DateTime, nullable=False, default=datetime.utcnow)


class Misinformation(base_cls):
    __tablename__ = 'misinformation'
    text = Column(String, primary_key=True)
    model_id = Column(String, ForeignKey('model.id'), primary_key=True)
    label_id = Column(String, ForeignKey('label.id'), nullable=False)
    date = Column(DateTime, nullable=False, default=datetime.utcnow)


class Source(base_cls):
    __tablename__ = 'source'
    type = Column(String, primary_key=True)
    date = Column(DateTime, nullable=False, default=datetime.utcnow)


class Input(base_cls):
    __tablename__ = 'input'
    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(String, nullable=False)
    source_type = Column(String, ForeignKey('source.type'), nullable=False)
    source_id = Column(String, nullable=False)
    misc = Column(JSON, nullable=True)
    date = Column(DateTime, nullable=False, default=datetime.utcnow)


class Output(base_cls):
    __tablename__ = 'output'
    id = Column(Integer, primary_key=True, autoincrement=True)
    input_id = Column(Integer, ForeignKey('input.id'), nullable=False)
    model_id = Column(String, ForeignKey('model.id'), nullable=False)
    prediction = Column(String, ForeignKey('label.id'), nullable=False)
    confidence = Column(Float, nullable=False)
    misc = Column(JSON, nullable=True)
    date = Column(DateTime, nullable=False, default=datetime.utcnow)
