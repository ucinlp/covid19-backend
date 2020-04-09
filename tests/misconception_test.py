from unittest import TestCase

from backend.ml.misconception import Misconception, MisconceptionDataset


class MisconceptionDatasetTest(TestCase):
    def test_from_jsonl(self):
        with open('tests/fixtures/misconceptions.jsonl', 'r') as f:
            misconception_list = MisconceptionDataset.from_jsonl(f)
        assert len(misconception_list) == 1
        expected = Misconception(
            sentence="don't lick faces",
            link="https://www.google.com"
        )
        assert misconception_list[0] == expected

    def test_hashable(self):
        misconceptions = (Misconception("don't read books", "https://www.google.com"),)
        misconception_dataset = MisconceptionDataset(misconceptions)
        hash(misconception_dataset)
