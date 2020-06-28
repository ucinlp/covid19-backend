from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import select

from backend.stream.db.table import get_table_class, Input, Output, Misinformation


def add_records(record_dicts, engine, table_class_name, returns_id=False):
    table_cls = get_table_class(table_class_name)
    table_cls.metadata.create_all(bind=engine, checkfirst=True)
    session = sessionmaker(bind=engine)()
    inserted_record_dict_list = list()
    inserted_id_list = list()
    try:
        record_list = list()
        for record_dict in record_dicts:
            record = table_cls(**record_dict)
            if not table_cls.check_if_exists(record, session):
                inserted_record_dict_list.append(record_dict)
                record_list.append(record)

        session.add_all(record_list)
        session.commit()
        if returns_id:
            session.flush()
            for r in record_list:
                session.refresh(r)
                inserted_id_list.append(r.id)
    except SQLAlchemyError as e:
        print(e)
    finally:
        session.close()

    if returns_id:
        return inserted_record_dict_list, inserted_id_list
    return inserted_record_dict_list


def select_all(engine, connection, table_class_name):
    table_cls = get_table_class(table_class_name)
    table_cls.metadata.create_all(bind=engine, checkfirst=True)
    results = None
    try:
        results = connection.execute(select([table_cls]))
    except SQLAlchemyError as e:
        print(e)
    return results


def select_all_input_output_pairs(engine, session):
    Input.metadata.create_all(bind=engine, checkfirst=True)
    Output.metadata.create_all(bind=engine, checkfirst=True)
    results = None
    try:
        results = session.query(Input, Output).filter(Input.id == Output.input_id).all()
    except SQLAlchemyError as e:
        print(e)
    return results


def get_inputs(engine, connection, source=None):
    Input.metadata.create_all(bind=engine, checkfirst=True)
    results = None
    try:
        if source is not None:
            results = connection.execute(select([Input]).where(Input.source_type == source))
        else:
            results = connection.execute(select([Input]))
    except SQLAlchemyError as e:
        print(e)
    return results


def get_misinfo(engine, connection, source=None):
    Misinformation.metadata.create_all(bind=engine, checkfirst=True)
    results = None
    try:
        if source is not None:
            results = connection.execute(select([Misinformation]).where(Misinformation.source == source))
        else:
            results = connection.execute(select([Misinformation]))
    except SQLAlchemyError as e:
        print(e)
    return results


def put_outputs(record_dicts, engine):
    input_size = len(record_dicts)
    inserted_record_dicts = add_records(record_dicts, engine, 'Output')
    output_size = len(inserted_record_dicts)
    if output_size != input_size:
        print('{} records were given, and {} of them were successfully inserted'.format(input_size, output_size))
        return False
    return True
