from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from backend.stream.db.table import get_table_class

base_cls = declarative_base()


def add_entities(entity_dicts, engine, table_class_name, table_name=None):
    if table_name is None:
        table_name = table_class_name.lower()

    table_cls = get_table_class(table_class_name)
    base_cls.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    entity_list = list()
    for entity_dict in entity_dicts:
        entity = table_cls(**entity_dict)
        if table_cls.check_if_exists(entity, session):
            entity_list.append(entity)
    try:
        session.add_all(entity_list)
        session.commit()
    except SQLAlchemyError as e:
        print(e)
    finally:
        session.close()
    return entity_list
