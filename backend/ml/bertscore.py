"""
BERTScore-based model
"""
from dataclasses import dataclass
from typing import List

import numpy as np
from overrides import overrides
import torch
from transformers import AutoModel, AutoTokenizer

from backend.ml.detector import Detector


@dataclass
class MaskedEmbeddings:
    embeddings: torch.FloatTensor
    mask: torch.FloatTensor


def soft_precision(scores: torch.FloatTensor,
                   mask: torch.FloatTensor) -> torch.FloatTensor:
    """
    Helper function for computing soft precision in batch.

    # Parameters
    scores : torch.FloatTensor
        Tensor of scores with shape: (num_refs, num_cands, max_ref_len, max_cand_len)
    mask : torch.FloatTensor
        Mask for the candidate tensor with shape: (num_cands, max_cand_len)
    """
    max_scores, _ = scores.max(dim=-2)
    masked_max_scores = max_scores * mask.unsqueeze(dim=0)
    precision = masked_max_scores.sum(dim=-1) / mask.sum(dim=-1).view(1, -1)
    return precision


def soft_recall(scores: torch.FloatTensor,
                mask: torch.FloatTensor) -> torch.FloatTensor:
    """
    Helper function for computing soft recall in batch.

    # Parameters
    scores : torch.FloatTensor
        Tensor of scores with shape: (num_refs, num_cands, max_ref_len, max_cand_len)
    mask : torch.FloatTensor
        Mask for the reference tensor with shape: (num_refs, max_ref_len)
    """
    max_scores, _ = scores.max(dim=-1)
    masked_max_scores = max_scores * mask.unsqueeze(dim=1)
    recall = masked_max_scores.sum(dim=-1) / mask.sum(dim=-1).view(-1, 1)
    return recall


def bertscore(candidate: MaskedEmbeddings,
              reference: MaskedEmbeddings) -> torch.FloatTensor:
    """
    BERTScore implementation.

    # Parameters
    candidate : MaskedEmbeddings
        Masked embeddings of the candidate sentences. The expected shape of the embedding tensor
        is: (num_cands, max_cand_len, embedding_dim)
    reference : MaskedEmbeddings
        Masked embeddings of the reference sentences. The expected shape of the embedding tensor
        is: (num_refs, max_ref_len, embedding_dim)
    """
    if candidate.embeddings.size(-1) != reference.embeddings.size(-1):
        raise ValueError("Embedding dimensions must match")

    # normalize embeddings
    candidate_embeddings = torch.nn.functional.normalize(candidate.embeddings, dim=-1)
    reference_embeddings = torch.nn.functional.normalize(reference.embeddings, dim=-1)

    # compute dot products between all contextualized word embeddings.
    # shape: (num_refs, num_cands, max_ref_len, max_cand_len)
    dot_products = torch.einsum('ire,jce->ijrc', reference_embeddings, candidate_embeddings)

    # compute soft precision and recall scores by taking max along reference and candidate sequence
    # length dimensions, respectively, and summing.
    # shape: (num_refs, num_cands)
    precision = soft_precision(dot_products, candidate.mask)
    recall = soft_recall(dot_products, reference.mask)
    f1 = 2 * precision * recall / (precision + recall)

    return f1


class BertScoreDetector(Detector):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)

    @overrides
    def _encode(self, sentences: List[str]) -> MaskedEmbeddings:
        model_input = self._tokenizer.batch_encode_plus(
            sentences,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        mask = model_input['attention_mask']
        embeddings, *_ = self._model(**model_input)
        return MaskedEmbeddings(embeddings, mask)

    @overrides
    def _score(self,
               encoded_sentences: MaskedEmbeddings,
               encoded_misconceptions: MaskedEmbeddings) -> np.ndarray:
        with torch.no_grad():
            score = bertscore(encoded_sentences, encoded_misconceptions)
        return score.cpu().numpy()
