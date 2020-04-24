"""
Abstract base class for misconception detectors
"""
from typing import Any, Dict, List, Union

import numpy as np

from backend.ml.misconception import MisconceptionDataset


# TODO: @rloganiv - The detector/misconception interaction would be better handled by using the
# observer design pattern to update the misconception encodings whenever the misconception dataset
# issues a notification that it has been updated. The issue with the current approach is that it
# may cache redundant data.
class Detector:
    """
    Abstract base class for a misconception detector.
    """
    def __init__(self):
        self._cache: Dict[str, Any] = {}

    def _encode(self, sentences: List[str]):
        raise NotImplementedError

    def _predict(self, scores: np.ndarray) -> List[List[int]]:
        raise NotImplementedError

    def predict(self,
                sentence: str,
                misconceptions: MisconceptionDataset) -> Dict[str, Any]:
        """
        Predicts whether a given piece of text corresponds to a misconception.

        # Parameters
        sentences : str or List[str]
            The text to score. Can either be a string or a list of strings.
        misconceptions : MisconceptionDataset
            The misconceptions to score against.

        # Returns
        output : Dict[str, Any]
            A dictionary of the prediction results. Will be serialized to JSON.
        """
        # TODO: @rloganiv - Current implementation works on a single sentence since that is the
        # expected input from the frontend. For evaluation purposes it probably makes sense to
        # allow predictions on batches on instances as well...maybe as a seperate method so output
        # type is consistent.
        scores = self.score(sentence, misconceptions)
        predictions = self._predict(scores)[0]

        # TODO: Relevance prediction
        readable_predictions = []
        for idx in predictions:
            score = scores[0, idx]
            misconception = misconceptions[idx]
            readable_predictions.append({
                'misinformation_score': float(score),
                'misinformation': misconception,
            })

        output_dict = {
            'input': sentence,
            'relevant': True,
            'predictions': readable_predictions,
        }
        return output_dict

    def _score(self, sentences, misconceptions) -> np.ndarray:
        raise NotImplementedError

    def score(self,
              sentences: Union[str, List[str]],
              misconceptions: MisconceptionDataset) -> np.ndarray:
        """
        Scores whether or not a given piece of text corresponds to a misconception.

        # Parameters
        sentences : str or List[str]
            The text to score. Can either be a string or a list of strings.
        misconceptions : MisconceptionDataset
            The misconceptions to score against.

        # Returns
        scores : np.ndarray
            An array with shape (num_sentences, num_misconceptions) containing the scores.
        """
        # encode misconceptions and cache to avoid needless recomputation.
        if misconceptions not in self._cache:
            self._cache[misconceptions] = self._encode(misconceptions.sentences)
        encoded_misconceptions = self._cache[misconceptions]

        if isinstance(sentences, str):
            sentences = [sentences]
        encoded_sentences = self._encode(sentences)

        scores = self._score(encoded_sentences, encoded_misconceptions)
        return scores

    def refresh_cache(self) -> None:
        """
        Refresh (clear) the misconception cache.
        """
        self._cache = {}
