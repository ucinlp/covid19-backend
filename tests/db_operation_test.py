from unittest import TestCase

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.stream.db.operation import add_entities
from backend.stream.db.table import Label, Reference, Model, Source, Misinformation, Input, Output


class OperationTest(TestCase):
    def test_add_labels(self):
        engine = create_engine('sqlite:///:memory:', echo=False)
        entity_dict_list = [{'id': 0, 'name': 'misleading'}]
        first_entities = add_entities(entity_dict_list, engine, 'Label')
        assert len(first_entities) == len(entity_dict_list)

        entity_dict_list.append({'id': 1, 'name': 'no scientific evidence'})
        second_entities = add_entities(entity_dict_list, engine, 'Label')
        assert len(second_entities) == len(entity_dict_list) - len(first_entities)

        session = sessionmaker(bind=engine)()
        assert all(Label.check_if_exists(entity, session) for entity in first_entities + second_entities)
        session.close()

    def test_add_references(self):
        engine = create_engine('sqlite:///:memory:', echo=False)
        entity_dict_list = [{'url': 'https://github.com'}]
        first_entities = add_entities(entity_dict_list, engine, 'Reference')
        assert len(first_entities) == len(entity_dict_list)

        entity_dict_list.append({'url': 'https://google.com'})
        second_entities = add_entities(entity_dict_list, engine, 'Reference')
        assert len(second_entities) == len(entity_dict_list) - len(first_entities)

        session = sessionmaker(bind=engine)()
        assert all(Reference.check_if_exists(entity, session) for entity in first_entities + second_entities)
        session.close()

    def test_add_models(self):
        engine = create_engine('sqlite:///:memory:', echo=False)
        entity_dict_list = [{'id': 'bert-test-ver', 'name': 'BERT score', 'config': {}}]
        first_entities = add_entities(entity_dict_list, engine, 'Model')
        assert len(first_entities) == len(entity_dict_list)

        entity_dict_list.append({'id': 'bert-prod-ver', 'name': 'BERT score', 'config': {}})
        second_entities = add_entities(entity_dict_list, engine, 'Model')
        assert len(second_entities) == len(entity_dict_list) - len(first_entities)

        session = sessionmaker(bind=engine)()
        assert all(Model.check_if_exists(entity, session) for entity in first_entities + second_entities)
        session.close()

    def test_add_sources(self):
        engine = create_engine('sqlite:///:memory:', echo=False)
        entity_dict_list = [{'type': 'Twitter'}]
        first_entities = add_entities(entity_dict_list, engine, 'Source')
        assert len(first_entities) == len(entity_dict_list)

        entity_dict_list.append({'type': 'Website'})
        second_entities = add_entities(entity_dict_list, engine, 'Source')
        assert len(second_entities) == len(entity_dict_list) - len(first_entities)

        session = sessionmaker(bind=engine)()
        assert all(Source.check_if_exists(entity, session) for entity in first_entities + second_entities)
        session.close()

    def test_add_misinformation(self):
        engine = create_engine('sqlite:///:memory:', echo=False)
        # Register a model
        add_entities([{'id': 'bert-test-ver', 'name': 'BERT score', 'config': {}}], engine, 'Model')
        # Register a label
        add_entities([{'id': 0, 'name': 'misleading'}], engine, 'Label')

        entity_dict_list = [{'text': 'no idea', 'model_id': 'bert-test-ver', 'label_id': 0}]
        first_entities = add_entities(entity_dict_list, engine, 'Misinformation')
        assert len(first_entities) == len(entity_dict_list)

        entity_dict_list.append({'text': 'hmm', 'model_id': 'bert-test-ver', 'label_id': 0})
        second_entities = add_entities(entity_dict_list, engine, 'Misinformation')
        assert len(second_entities) == len(entity_dict_list) - len(first_entities)

        session = sessionmaker(bind=engine)()
        assert all(Misinformation.check_if_exists(entity, session) for entity in first_entities + second_entities)
        session.close()

    def test_add_inputs(self):
        engine = create_engine('sqlite:///:memory:', echo=False)
        # Register a source
        add_entities([{'type': 'Twitter'}], engine, 'Source')

        entity_dict_list = [{'text': 'no idea', 'source_type': 'Twitter', 'source_id': 'some tweet ID'}]
        first_entities = add_entities(entity_dict_list, engine, 'Input')
        assert len(first_entities) == len(entity_dict_list)

        entity_dict_list.append({'text': 'still no idea', 'source_type': 'Twitter', 'source_id': 'some tweet ID'})
        second_entities = add_entities(entity_dict_list, engine, 'Input')
        assert len(second_entities) == len(entity_dict_list) - len(first_entities)

        session = sessionmaker(bind=engine)()
        assert all(Input.check_if_exists(entity, session) for entity in first_entities + second_entities)
        session.close()

    def test_add_outputs(self):
        engine = create_engine('sqlite:///:memory:', echo=False)
        # Register a source
        add_entities([{'type': 'Twitter'}], engine, 'Source')
        # Register an input
        add_entities([{'text': 'no', 'source_type': 'Twitter', 'source_id': 'some tweet ID'}], engine, 'Input')
        add_entities([{'text': 'hmm', 'source_type': 'Twitter', 'source_id': 'some tweet ID'}], engine, 'Input')
        # Register a model
        add_entities([{'id': 'bert-test-ver', 'name': 'BERT score', 'config': {}}], engine, 'Model')
        # Register a label
        add_entities([{'id': 0, 'name': 'misleading'}], engine, 'Label')

        entity_dict_list = [{'input_id': 1, 'model_id': 'bert-test-ver', 'prediction': 0, 'confidence': 0.5}]
        first_entities = add_entities(entity_dict_list, engine, 'Output')
        assert len(first_entities) == len(entity_dict_list)

        entity_dict_list.append({'input_id': 2, 'model_id': 'bert-test-ver', 'prediction': 0, 'confidence': 0.5})
        second_entities = add_entities(entity_dict_list, engine, 'Input')
        assert len(second_entities) == len(entity_dict_list) - len(first_entities)

        session = sessionmaker(bind=engine)()
        assert all(Output.check_if_exists(entity, session) for entity in first_entities + second_entities)
        session.close()
