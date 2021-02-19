import logging
from unittest import TestCase

import torch
from overrides import overrides

from backend.ml.bertscore import BertScoreDetector
from backend.ml.pipeline import Pipeline
from backend.ml.sentence_bert import SentenceBertClassifier
from backend.ml.misconception import MisconceptionDataset

logger = logging.getLogger(__name__)


def test_pipeline():
    # TODO: Periodically reload.
    logger.info('Loading misconceptions')
    with open('misconceptions.jsonl', 'r') as f:
        misconceptions = MisconceptionDataset.from_jsonl(f)

    logger.info('Loading models')
    retriever = BertScoreDetector('digitalepidemiologylab/covid-twitter-bert')
    detector = SentenceBertClassifier(
        model_name='digitalepidemiologylab/covid-twitter-bert',
        num_classes=3,
    )
    state_dict = torch.load('/home/rlogan/SBERT-MNLI-ckpt-2.pt')
    logger.info('Restoring detector checkpoint')
    detector.load_state_dict(state_dict)
    pipeline = Pipeline(retriever=retriever, detector=detector)

    pipeline("North Korea and China conspired together to create the coronavirus.", misconceptions)

