"""
SentenceBERT-based model
"""
from typing import Any, Dict, List

import numpy as np
from overrides import overrides
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from backend.ml.detector import Detector


def cosine_similarity(x: torch.FloatTensor,
                      y: torch.FloatTensor) -> torch.FloatTensor:
    """
    Measures the pairwise cosine similarity between two tensors.

    # Parameters
    x : torch.FloatTensor
        Input tensor of shape (num_cands, embedding_dim)
    y: torch.FloatTensor
        Input tensor of shape (num_refs, embedding_dim)

    # Returns
    scores : torch.FloatTensor
        Tensor of pairwise cosine similarity scores, with shape: (num_cands, num_refs)
    """
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    scores = torch.einsum('ce,re->cr', x, y)
    return scores


class SentenceBertDetector(Detector, torch.nn.Module):
    def __init__(self, model_name: str) -> None:
        torch.nn.Module.__init__(self)
        Detector.__init__(self)
        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)

    @overrides
    def forward(self,
                anchor_sentences: List[str],
                positive_sentences: List[str],
                negative_sentences: List[str],
                loss_kwargs: Dict[str, Any] = None) -> torch.FloatTensor:
        """
        Returns triplet margin loss so that model can be fine-tuned for ranking.

        # Parameters
        anchor_sentences: List[str]
            List of "anchor" sentences.
        positive_sentences: List[str]
            List of sentences that should be close to corresponding "anchor" sentences.
        negative_sentences: List[str]
            List of sentences that should be distant to corresponding "anchor" sentences.
        loss_kwargs: Dict[str, Any]
            Optional dictionary of arguments to be passed to the `triplet_margin_loss` function.
            See PyTorch docs for more details:
                https://pytorch.org/docs/master/nn.functional.html#triplet-margin-loss

        # Returns
        loss : torch.FloatTensor
            Scalar loss value.
        """
        assert len(anchor_sentences) == len(positive_sentences) == len(negative_sentences)
        loss_kwargs = loss_kwargs or {}
        anchor_embeddings = self._encode(anchor_sentences)
        positive_embeddings = self._encode(positive_sentences)
        negative_embeddings = self._encode(negative_sentences)
        loss = F.triplet_margin_loss(
            anchor_embeddings,
            positive_embeddings,
            negative_embeddings,
            **loss_kwargs
        )
        return loss

    @overrides
    def _encode(self, sentences: List[str]) -> torch.FloatTensor:
        model_input = self._tokenizer.batch_encode_plus(
            sentences,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        mask = model_input['attention_mask']
        embeddings, *_ = self._model(**model_input)
        masked_embeddings = embeddings * mask.unsqueeze(-1)
        pooled_embeddings = masked_embeddings.mean(1)  # average over sequence dim
        return pooled_embeddings

    @overrides
    def _score(self,
               encoded_sentences: torch.FloatTensor,
               encoded_misconceptions: torch.FloatTensor) -> np.ndarray:
        with torch.no_grad():
            score = cosine_similarity(encoded_sentences, encoded_misconceptions)
        return score.cpu().numpy()
