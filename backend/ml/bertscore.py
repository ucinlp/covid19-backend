"""
BERTScore-based model
"""
from dataclasses import dataclass
from typing import List

import numpy as np
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

    # Returns
    output_dict : Dictionary
        f1 : torch.FloatTensor
            Tensor of pairwise f1 scores, with shape: (num_cands, num_refs)
        precision : torch.FloatTensor
            Tensor of the soft precision scores, with shape: (num_cands, num_refs)
        recall : torch.FloatTensor
            Tensor of the soft recall scores, with shape: (num_cands, num_refs)
    """
    if candidate.embeddings.size(-1) != reference.embeddings.size(-1):
        raise ValueError("Embedding dimensions must match")

    # Normalize embeddings.
    candidate_embeddings = torch.nn.functional.normalize(candidate.embeddings, dim=-1)
    reference_embeddings = torch.nn.functional.normalize(reference.embeddings, dim=-1)

    # Compute dot products between all contextualized word embeddings.
    # shape: (num_refs, num_cands, max_ref_len, max_cand_len)
    dot_products = torch.einsum('ire,jce->ijrc', reference_embeddings, candidate_embeddings)

    # We assign scores of pad tokens to a large negative value to prevent them from being matched.
    mask = torch.einsum('ir,jc->ijrc', reference.mask, candidate.mask)
    dot_products = dot_products - 1e13 * (1 - mask)

    # Transpose the tensors since intuitively the `num_cands` dimension (e.g., batch size) should
    # be first. We opted not to do this earlier to make the above computations match the equations
    # in the BERTScore paper.
    # shape: (num_refs, num_cands)
    precision = soft_precision(dot_products, candidate.mask).transpose(0, 1)
    recall = soft_recall(dot_products, reference.mask).transpose(0, 1)
    f1 = 2 * precision * recall / (precision + recall)

    output_dict = {'precision': precision, 'recall': recall, 'f1': f1}

    return output_dict


class BertScoreDetector(Detector, torch.nn.Module):
    """
    BERTScore Detector

    # Parameters
    model_name : `str`
        Name of model passed to HuggingFace's AutoModel/AutoTokenizer.
    score_type : `str`, default='f1'
        Optional. Type of score to return, one of: 'precision', 'recall', 'f1'.
    """
    def __init__(self,
                 model_name: str,
                 score_type: str = 'f1') -> None:
        Detector.__init__(self)
        torch.nn.Module.__init__(self)
        if score_type not in ('precision', 'recall', 'f1'):
            raise ValueError(f'Invalid score type "{score_type}."')
        self._model_name = model_name
        self._score_type = score_type
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)

    def _encode(self, sentences: List[str]) -> MaskedEmbeddings:
        device = next(self.parameters()).device
        model_input = self._tokenizer.batch_encode_plus(
            sentences,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        model_input = {k: v.to(device) for k, v in model_input.items()}
        mask = model_input['attention_mask']
        embeddings, *_ = self._model(**model_input)
        return MaskedEmbeddings(embeddings, mask)

    def _score(self,
               sentences: MaskedEmbeddings,
               misconceptions: MaskedEmbeddings) -> np.ndarray:
        with torch.no_grad():
            score_dict = bertscore(sentences, misconceptions)
        score = score_dict[self._score_type]
        return score.cpu().numpy()

    def _predict(self, scores: np.ndarray) -> List[List[int]]:
        # TODO: @rloganiv - Something smarter thank just returning top-5.
        k = 5
        topk = scores.argsort(axis=-1)[:, :-(k+1):-1]
        return topk.tolist()
