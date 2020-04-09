from unittest import TestCase

import torch

from backend.bertscore import (
    bertscore, soft_precision, soft_recall, BertScoreDetector, MaskedEmbeddings
)
from backend.misconception import Misconception, MisconceptionDataset


SMALL_MODEL_IDENTIFIER = "julien-c/bert-xsmall-dummy"


class BertScoreTest(TestCase):
    def test_output_is_correct(self):
        # tests score computation is correct for reference and candidates comprised of two
        # identical sequences:
        #   sequence 1: contains two tokens
        #   sequence 2: contains three tokens
        embeddings = torch.FloatTensor([
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        ])
        mask = torch.FloatTensor([[1, 1, 0], [1, 1, 1]])
        candidate = MaskedEmbeddings(embeddings, mask)
        reference = candidate
        score = bertscore(candidate, reference)
        expected_score = torch.FloatTensor([
            [1, .8],
            [.8, 1]
        ])
        assert torch.allclose(score, expected_score)

    def test_soft_precision_and_recall(self):
        # tests that precision is 2/3 and recall is 1 for a reference sequence of length two and a
        # candidate sequence of length three, where two of the tokens perfectly match.
        scores = torch.FloatTensor([[1, 0, 0], [0, 1, 0]]).view(1, 1, 2, 3)
        candidate_mask = torch.FloatTensor([[1, 1, 1]])
        reference_mask = torch.FloatTensor([[1, 1]])

        precision = soft_precision(scores, candidate_mask)
        expected_precision = torch.FloatTensor([[2/3]])
        assert torch.allclose(precision, expected_precision)

        recall = soft_recall(scores, reference_mask)
        expected_recall = torch.FloatTensor([[1]])
        assert torch.allclose(recall, expected_recall)

    def test_fails_on_bad_input(self):
        candidate = MaskedEmbeddings(torch.randn(1, 1, 2), torch.ones(1, 1))
        reference = MaskedEmbeddings(torch.randn(1, 1, 3), torch.ones(1, 1))
        with self.assertRaises(ValueError):
            bertscore(candidate, reference)


class BertScoreDetectorTest(TestCase):
    def test_score(self):
        # checks that we can load a small transformer and use it to score test sentences without
        # raising any errors.
        misconceptions = MisconceptionDataset((
            Misconception('glue is good for digestion', 'https://bogushealth.org'),
        ))
        sentences = ['Lorem ipsum', 'dolor sit amet']
        detector = BertScoreDetector(SMALL_MODEL_IDENTIFIER)
        detector.score(sentences, misconceptions)