from unittest import TestCase

import numpy as np
import torch

from backend.ml.misconception import MisconceptionDataset
from backend.ml.sentence_bert import cosine_similarity
from backend.ml.sentence_bert import SentenceBertBase


#SMALL_MODEL_IDENTIFIER = "julien-c/bert-xsmall-dummy"
SMALL_MODEL_IDENTIFIER = "roberta-base"


class CosineSimilarityTest(TestCase):
    def test_output_is_corect(self):
        # Tests that score of a batch of orthogonal tensors against itself is the identity matrix.
        x = torch.FloatTensor([
            [1,  1],
            [1, -1],
        ])
        score = cosine_similarity(x, x)
        expected_score = torch.FloatTensor([
            [1,  0],
            [0,  1]
        ])
        assert torch.allclose(score, expected_score)


class SentenceBertBaseTest(TestCase):
    def setUp(self):
        self.detector = SentenceBertBase(SMALL_MODEL_IDENTIFIER)

    def test_scores_not_affected_by_padding(self):
        with open('tests/fixtures/misconceptions.jsonl', 'r') as f:
            misconceptions = MisconceptionDataset.from_jsonl(f)
        sentences = ['Lorem ipsum', 'dolor sit amet']
        score_a = self.detector.score(sentences[0], misconceptions)
        score_b = self.detector.score(sentences, misconceptions)[0]
        assert np.allclose(score_a, score_b)
