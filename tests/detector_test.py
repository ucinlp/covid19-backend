from unittest import TestCase

import numpy as np
from overrides import overrides

from backend.ml.detector import Detector
from backend.ml.misconception import MisconceptionDataset


class MockDetector(Detector):
    @overrides
    def _encode(self, sentences):
        return sentences

    @overrides
    def _score(self, encoded_sentences, encoded_misconceptions):
        num_sentences = len(encoded_sentences)
        num_misconceptions = len(encoded_misconceptions)
        return np.zeros((num_sentences, num_misconceptions))

    @overrides
    def _predict(self, scores):
        return [[0]]


class DetectorTest(TestCase):
    def setUp(self):
        self.detector = MockDetector()
        with open('tests/fixtures/misconceptions.jsonl', 'r') as f:
            misconceptions = MisconceptionDataset.from_jsonl(f)
        self.misconceptions = misconceptions

    def test_score_caches_misconceptions(self):
        sentences = ['Lorem ipsum', 'dolor sit amet']
        self.detector.score(sentences, self.misconceptions)
        self.assertIn(self.misconceptions, self.detector._cache)
        self.assertListEqual(self.misconceptions.sentences,
                             self.detector._cache[self.misconceptions])

    def test_predict(self):
        sentence = 'Lorem ipsum'
        output_dict = self.detector.predict(sentence, self.misconceptions)
        self.assertEqual(output_dict['input'], sentence)
        self.assertTrue(output_dict['relevant'])
        expected_predictions = [{
            'misinformation_score': 0.0,
            'misinformation': self.misconceptions[0],
        }]
        self.assertListEqual(output_dict['predictions'], expected_predictions)
