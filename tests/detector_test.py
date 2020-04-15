from unittest import TestCase

import numpy as np
from overrides import overrides

from backend.ml.detector import Detector
from backend.ml.misconception import Misconception, MisconceptionDataset


class MockDetector(Detector):
    @overrides
    def _encode(self, sentences):
        return sentences

    @overrides
    def _score(self, encoded_sentences, encoded_misconceptions):
        num_sentences = len(encoded_sentences)
        num_misconceptions = len(encoded_misconceptions)
        return np.zeros((num_sentences, num_misconceptions))


class DetectorTest(TestCase):
    def test_score_caches_misconceptions(self):
        with open('tests/fixtures/misconceptions.jsonl', 'r') as f:
            misconceptions = MisconceptionDataset.from_jsonl(f)
        sentences = ['Lorem ipsum', 'dolor sit amet']
        detector = MockDetector()
        detector.score(sentences, misconceptions)
        self.assertIn(misconceptions, detector._cache)
        self.assertListEqual(detector._cache[misconceptions], misconceptions.sentences)
