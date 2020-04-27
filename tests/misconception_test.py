from unittest import TestCase

from backend.ml.misconception import Misconception, MisconceptionDataset


class MisconceptionDatasetTest(TestCase):
    def test_from_jsonl(self):
        with open('tests/fixtures/misconceptions.jsonl', 'r') as f:
            misconceptions = MisconceptionDataset.from_jsonl(f)
        assert len(misconceptions) == 1
        expected = Misconception(
            id=1,
            canonical_sentence="Don't lick faces",
            sources=("https://www.google.com",),
            category=tuple(),
            pos_variations=("Don't lick faces",),
            neg_variations=tuple(),
            reliability_score=1,
            origin="Sameer"
        )
        assert misconceptions[0] == expected

    def test_hashable(self):
        misconceptions = MisconceptionDataset(tuple(), tuple(), uid=123)
        assert hash(misconceptions) == 123
