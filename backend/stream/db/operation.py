from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import select

from backend.stream.db.table import get_table_class, Input, Output


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


def select_all(engine, table_class_name):
    table_cls = get_table_class(table_class_name)
    table_cls.metadata.create_all(bind=engine, checkfirst=True)
    session = sessionmaker(bind=engine)()
    results, connection = None, None
    try:
        connection = engine.connect()
        results = connection.execute(select([table_cls]))
        connection.close()
    except SQLAlchemyError as e:
        print(e)
    finally:
        if connection is not None:
            session.close()
    return results


def select_all_input_output_pairs(engine):
    Input.metadata.create_all(bind=engine, checkfirst=True)
    Output.metadata.create_all(bind=engine, checkfirst=True)
    session = sessionmaker(bind=engine)()
    results, connection = None, None
    try:
        connection = engine.connect()
        results = session.query(Input, Output).filter(Input.id == Output.input_id).all()
        connection.close()
    except SQLAlchemyError as e:
        print(e)
    finally:
        if connection is not None:
            session.close()
    return results


def get_inputs(engine, source=None):
    Input.metadata.create_all(bind=engine, checkfirst=True)
    session = sessionmaker(bind=engine)()
    results, connection = None, None
    try:
        connection = engine.connect()
        if source is not None:
            results = connection.execute(select([Input]).where(Input.source_type == source))
        else:
            results = connection.execute(select([Input]))
        connection.close()
    except SQLAlchemyError as e:
        print(e)
    finally:
        if connection is not None:
            session.close()
    return results
