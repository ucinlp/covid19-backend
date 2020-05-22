from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from backend.stream.db.table import get_table_class


def add_entities(entity_dicts, engine, table_class_name):
    table_cls = get_table_class(table_class_name)
    table_cls.metadata.create_all(bind=engine, checkfirst=True)
    session = sessionmaker(bind=engine)()
    entity_list = list()
    try:
        for entity_dict in entity_dicts:
            entity = table_cls(**entity_dict)
            if not table_cls.check_if_exists(entity, session):
                entity_list.append(entity)
                
        session.add_all(entity_list)
        session.commit()
    except SQLAlchemyError as e:
        print(e)
    finally:
        session.close()
    return entity_list
