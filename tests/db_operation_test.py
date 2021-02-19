import json
from unittest import TestCase

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.stream.db.operation import add_records, select_all, select_all_input_output_pairs, get_inputs, \
    get_misinfo, put_outputs, select_input_output_pairs_by_date, select_recent_input_output_pairs
from backend.stream.db.table import Label, Reference, Model, Source, Misinformation, Input, Output


def check_if_registered(table_cls, entity_dict_list, session):
    expected_entities = [table_cls(**entity_dict) for entity_dict in entity_dict_list]
    return all(table_cls.check_if_exists(entity, session) for entity in expected_entities)


class OperationTest(TestCase):
    def test_add_labels(self):
        engine = create_engine('sqlite://', echo=False)
        entity_dict_list = [{'id': 0, 'name': 'misleading'}]
        first_entities = add_records(entity_dict_list, engine, 'Label')
        assert len(first_entities) == len(entity_dict_list)

        entity_dict_list.append({'id': 1, 'name': 'no scientific evidence'})
        second_entities = add_records(entity_dict_list, engine, 'Label')
        assert len(second_entities) == len(entity_dict_list) - len(first_entities)

        session = sessionmaker(bind=engine)()
        assert check_if_registered(Label, entity_dict_list, session)
        session.close()

    def test_add_references(self):
        engine = create_engine('sqlite://', echo=False)
        entity_dict_list = [{'url': 'https://github.com'}]
        first_entities = add_records(entity_dict_list, engine, 'Reference')
        assert len(first_entities) == len(entity_dict_list)

        entity_dict_list.append({'url': 'https://google.com'})
        second_entities = add_records(entity_dict_list, engine, 'Reference')
        assert len(second_entities) == len(entity_dict_list) - len(first_entities)

        session = sessionmaker(bind=engine)()
        assert check_if_registered(Reference, entity_dict_list, session)
        session.close()

    def test_add_models(self):
        engine = create_engine('sqlite://', echo=False)
        entity_dict_list = [{'id': 'bert-test-ver', 'name': 'BERT score', 'human': False, 'config': {}}]
        first_entities = add_records(entity_dict_list, engine, 'Model')
        assert len(first_entities) == len(entity_dict_list)

        entity_dict_list.append({'id': 'bert-prod-ver', 'name': 'BERT score', 'human': False, 'config': {}})
        second_entities = add_records(entity_dict_list, engine, 'Model')
        assert len(second_entities) == len(entity_dict_list) - len(first_entities)

        session = sessionmaker(bind=engine)()
        assert check_if_registered(Model, entity_dict_list, session)
        session.close()

    def test_add_sources(self):
        engine = create_engine('sqlite://', echo=False)
        entity_dict_list = [{'type': 'Twitter'}]
        first_entities = add_records(entity_dict_list, engine, 'Source')
        assert len(first_entities) == len(entity_dict_list)

        entity_dict_list.append({'type': 'Website'})
        second_entities = add_records(entity_dict_list, engine, 'Source')
        assert len(second_entities) == len(entity_dict_list) - len(first_entities)

        session = sessionmaker(bind=engine)()
        assert check_if_registered(Source, entity_dict_list, session)
        session.close()

    def test_add_misinformation(self):
        engine = create_engine('sqlite://', echo=False)
        # Register a model
        add_records([{'id': 'bert-test-ver', 'name': 'BERT score', 'human': False, 'config': {}}], engine, 'Model')
        # Register a label
        add_records([{'id': 0, 'name': 'misleading'}], engine, 'Label')

        entity_dict_list = [{'text': 'no idea', 'url': json.dumps({'list': ['http://www.cdc.gov']}), 'source': 'CDC',
                             'reliability': 3}]
        first_entities = add_records(entity_dict_list, engine, 'Misinformation')
        assert len(first_entities) == len(entity_dict_list)

        entity_dict_list.append({'text': 'hmm', 'url': json.dumps({'list': ['http://www.who.int']}), 'source': 'WHO',
                                 'reliability': 3})
        second_entities = add_records(entity_dict_list, engine, 'Misinformation')
        assert len(second_entities) == len(entity_dict_list) - len(first_entities)

        session = sessionmaker(bind=engine)()
        assert check_if_registered(Misinformation, entity_dict_list, session)
        session.close()

    def test_add_inputs(self):
        engine = create_engine('sqlite://', echo=False)
        # Register a source
        add_records([{'type': 'Twitter'}], engine, 'Source')

        entity_dict_list = [{'text': 'no idea', 'source_type': 'Twitter', 'source_id': 'some tweet ID'}]
        first_entities = add_records(entity_dict_list, engine, 'Input')
        assert len(first_entities) == len(entity_dict_list)

        session = sessionmaker(bind=engine)()
        entity_dict_list[0]['id'] = 1
        assert check_if_registered(Input, entity_dict_list, session)
        session.close()

    def test_add_outputs(self):
        engine = create_engine('sqlite://', echo=False)
        # Register a source
        add_records([{'type': 'Twitter'}], engine, 'Source')
        # Register an input
        add_records([{'text': 'no', 'source_type': 'Twitter', 'source_id': 'some tweet ID'}], engine, 'Input')
        add_records([{'text': 'hmm', 'source_type': 'Twitter', 'source_id': 'some tweet ID'}], engine, 'Input')
        # Register a model
        add_records([{'id': 'bert-test-ver', 'name': 'BERT score', 'human': False, 'config': {}}], engine, 'Model')
        # Register a label
        add_records([{'id': 0, 'name': 'misleading'}], engine, 'Label')
        # Register misinformation
        add_records([{'text': 'hmm', 'url': json.dumps({'list': ['http://www.who.int']}), 'source': 'WHO',
                       'reliability': 3}], engine, 'Misinformation')
        entity_dict_list = [{'input_id': 1, 'model_id': 'bert-test-ver', 'label_id': 0,
                             'misinfo_id': 1, 'confidence': 0.5},
                            {'input_id': 2, 'model_id': 'bert-test-ver', 'label_id': 0,
                             'misinfo_id': 1, 'confidence': 0.5}]
        first_entities = add_records(entity_dict_list, engine, 'Output')
        assert len(first_entities) == len(entity_dict_list)

        session = sessionmaker(bind=engine)()
        for i, entity_dict in enumerate(entity_dict_list):
            entity_dict['id'] = i + 1
        assert check_if_registered(Output, entity_dict_list, session)
        session.close()

    def test_select_all(self):
        engine = create_engine('sqlite://', echo=False)
        connection = engine.connect()
        entity_dict_list = [{'id': 0, 'name': 'misleading'}, {'id': 1, 'name': 'no scientific evidence'}]
        add_records(entity_dict_list, engine, 'Label')
        results = select_all(engine, connection, 'Label')
        keys = results.keys()
        values = [row.values() for row in results]
        assert keys == ['id', 'name', 'misc', 'date'] and len(values) == len(entity_dict_list)
        connection.close()

    def test_select_all_input_output_pairs(self):
        engine = create_engine('sqlite://', echo=False)
        # Register a source
        add_records([{'type': 'Twitter'}], engine, 'Source')
        # Register an input
        add_records([{'text': 'no', 'source_type': 'Twitter', 'source_id': 'some tweet ID'}], engine, 'Input')
        add_records([{'text': 'hmm', 'source_type': 'Twitter', 'source_id': 'some tweet ID'}], engine, 'Input')
        add_records([{'text': 'ok', 'source_type': 'Twitter', 'source_id': 'some tweet ID'}], engine, 'Input')
        # Register a model
        add_records([{'id': 'bert-test-ver', 'name': 'BERT score', 'human': False, 'config': {}}], engine, 'Model')
        # Register a label
        add_records([{'id': 0, 'name': 'misleading'}], engine, 'Label')
        # Register misinformation
        add_records([{'text': 'hmm', 'url': json.dumps({'list': ['http://www.who.int']}), 'source': 'WHO',
                       'reliability': 3}], engine, 'Misinformation')
        entity_dict_list = [{'input_id': 1, 'model_id': 'bert-test-ver', 'label_id': 0,
                             'misinfo_id': 1, 'confidence': 0.5},
                            {'input_id': 2, 'model_id': 'bert-test-ver', 'label_id': 0,
                             'misinfo_id': 1, 'confidence': 0.5}]
        add_records(entity_dict_list, engine, 'Output')
        session = sessionmaker(bind=engine)()
        results = select_all_input_output_pairs(engine, session)
        assert all(row.Input.id == row.Output.input_id for row in results)
        session.close()

    def test_get_inputs(self):
        engine = create_engine('sqlite://', echo=False)
        # Register a source
        add_records([{'type': 'Twitter'}, {'type': 'Website'}], engine, 'Source')

        entity_dict_list = [{'text': 'no idea', 'source_type': 'Twitter', 'source_id': 'some tweet ID'},
                            {'text': 'ok', 'source_type': 'Website', 'source_id': 'N/A'}]
        first_entities = add_records(entity_dict_list, engine, 'Input')
        assert len(first_entities) == len(entity_dict_list)

        connection = engine.connect()
        results = get_inputs(engine, connection)
        assert all(row.id == i + 1 for i, row in enumerate(results))
        results = get_inputs(engine, connection, source='Twitter')
        assert all(row.source_type == 'Twitter' for row in results)
        connection.close()

    def test_get_misinfo(self):
        engine = create_engine('sqlite://', echo=False)
        # Register a model
        add_records([{'id': 'bert-test-ver', 'name': 'BERT score', 'human': False, 'config': {}}], engine, 'Model')
        # Register a label
        add_records([{'id': 0, 'name': 'misleading'}], engine, 'Label')

        entity_dict_list = [{'text': 'no idea', 'url': json.dumps({'list': ['http://www.cdc.gov']}), 'source': 'CDC',
                             'reliability': 3},
                            {'text': 'hmm', 'url': json.dumps({'list': ['http://www.who.int']}), 'source': 'WHO',
                             'reliability': 3}]
        first_entities = add_records(entity_dict_list, engine, 'Misinformation')
        assert len(first_entities) == len(entity_dict_list)

        connection = engine.connect()
        results = get_misinfo(engine, connection)
        assert all(row.id == i + 1 for i, row in enumerate(results))
        results = get_misinfo(engine, connection, source='CDC')
        assert all(row.source == 'CDC' for row in results)
        connection.close()

    def test_put_outputs(self):
        engine = create_engine('sqlite://', echo=False)
        # Register a source
        add_records([{'type': 'Twitter'}], engine, 'Source')
        # Register an input
        add_records([{'text': 'no', 'source_type': 'Twitter', 'source_id': 'some tweet ID'}], engine, 'Input')
        add_records([{'text': 'hmm', 'source_type': 'Twitter', 'source_id': 'some tweet ID'}], engine, 'Input')
        # Register a model
        add_records([{'id': 'bert-test-ver', 'name': 'BERT score', 'human': False, 'config': {}}], engine, 'Model')
        # Register a label
        add_records([{'id': 0, 'name': 'misleading'}], engine, 'Label')
        # Register misinformation
        add_records([{'text': 'hmm', 'url': json.dumps({'list': ['http://www.who.int']}), 'source': 'WHO',
                       'reliability': 3}], engine, 'Misinformation')
        entity_dict_list = [{'input_id': 1, 'model_id': 'bert-test-ver', 'label_id': 0,
                             'misinfo_id': 1, 'confidence': 0.5},
                            {'input_id': 2, 'model_id': 'bert-test-ver', 'label_id': 0,
                             'misinfo_id': 1, 'confidence': 0.5}]
        assert put_outputs(entity_dict_list, engine)

    def test_select_input_output_pairs_by_date(self):
        engine = create_engine('sqlite://', echo=False)
        # Register a source
        add_records([{'type': 'Twitter'}], engine, 'Source')
        # Register an input
        add_records([{'text': 'no', 'source_type': 'Twitter', 'source_id': 'some tweet ID'}], engine, 'Input')
        add_records([{'text': 'hmm', 'source_type': 'Twitter', 'source_id': 'some tweet ID'}], engine, 'Input')
        add_records([{'text': 'ok', 'source_type': 'Twitter', 'source_id': 'some tweet ID'}], engine, 'Input')
        # Register a model
        add_records([{'id': 'bert-test-ver', 'name': 'BERT score', 'human': False, 'config': {}}], engine, 'Model')
        # Register a label
        add_records([{'id': 0, 'name': 'misleading'}], engine, 'Label')
        # Register misinformation
        add_records([{'text': 'hmm', 'url': json.dumps({'list': ['http://www.who.int']}), 'source': 'WHO',
                      'reliability': 3}], engine, 'Misinformation')
        entity_dict_list = [{'input_id': 1, 'model_id': 'bert-test-ver', 'label_id': 0,
                             'misinfo_id': 1, 'confidence': 0.5},
                            {'input_id': 2, 'model_id': 'bert-test-ver', 'label_id': 0,
                             'misinfo_id': 1, 'confidence': 0.5}]
        add_records(entity_dict_list, engine, 'Output')
        session = sessionmaker(bind=engine)()
        results = select_input_output_pairs_by_date(engine, session, from_date='1900-01-01', until_date='2100-01-01')
        assert all(row.Input.id == row.Output.input_id for row in results)
        session.close()

    def test_select_recent_input_output_pairs(self):
        engine = create_engine('sqlite://', echo=False)
        # Register a source
        add_records([{'type': 'Twitter'}], engine, 'Source')
        # Register an input
        add_records([{'text': 'no', 'source_type': 'Twitter', 'source_id': 'some tweet ID'}], engine, 'Input')
        add_records([{'text': 'hmm', 'source_type': 'Twitter', 'source_id': 'some tweet ID'}], engine, 'Input')
        add_records([{'text': 'ok', 'source_type': 'Twitter', 'source_id': 'some tweet ID'}], engine, 'Input')
        # Register a model
        add_records([{'id': 'bert-test-ver', 'name': 'BERT score', 'human': False, 'config': {}}], engine, 'Model')
        # Register a label
        add_records([{'id': 0, 'name': 'misleading'}], engine, 'Label')
        # Register misinformation
        add_records([{'text': 'hmm', 'url': json.dumps({'list': ['http://www.who.int']}), 'source': 'WHO',
                       'reliability': 3}], engine, 'Misinformation')
        entity_dict_list = [{'input_id': 1, 'model_id': 'bert-test-ver', 'label_id': 0,
                             'misinfo_id': 1, 'confidence': 0.5},
                            {'input_id': 2, 'model_id': 'bert-test-ver', 'label_id': 0,
                             'misinfo_id': 1, 'confidence': 0.5}]
        add_records(entity_dict_list, engine, 'Output')
        session = sessionmaker(bind=engine)()
        results = select_recent_input_output_pairs(engine, session, recent_k=1)
        assert len(results) == 1
        session.close()
