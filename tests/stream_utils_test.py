from unittest import TestCase

from backend.stream.common import file_util, misc_util


class FileUtilTest(TestCase):
    def test_get_dir_paths(self):
        expected_list = ['tests/fixtures']
        dir_paths = file_util.get_dir_paths('tests/')
        assert dir_paths == expected_list

    def test_get_file_paths(self):
        expected_list = ['tests/fixtures/misconceptions.jsonl']
        file_paths = file_util.get_file_paths('tests/fixtures/')
        assert file_paths == expected_list


class MiscUtilTest(TestCase):
    def test_overwrite_dict(self):
        org_dict = {'test': 'yay', 10: ['hello', 'world']}
        sub_dict = {'test': 'wow', '10': ['should not overwrite']}
        expected_dict = {'test': 'wow', 10: ['hello', 'world'], '10': ['should not overwrite']}
        misc_util.overwrite_dict(org_dict, sub_dict)
        assert org_dict == expected_dict

    def test_overwrite_config(self):
        config = {'test': 'yay', 10: ['hello', 'world']}
        json_str = '{"test": "wow", "10": ["should not overwrite"]}'
        expected_dict = {'test': 'wow', 10: ['hello', 'world'], '10': ['should not overwrite']}
        misc_util.overwrite_config(config, json_str)
        assert config == expected_dict
