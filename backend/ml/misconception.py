"""
Objects that load/store/validate misconception data.
"""
from dataclasses import dataclass
from hashlib import md5
from io import StringIO
import json
from typing import Tuple


def _tuplify(x):
    """Converts lists to tuples."""
    if isinstance(x, list):
        return tuple(x)
    else:
        return x


@dataclass(eq=True, frozen=True)
class Misconception:
    id: str
    canonical_sentence: str
    sources: Tuple[str]
    category: Tuple[str]
    pos_variations: Tuple[str]
    neg_variations: Tuple[str]
    reliability_score: int
    origin: str


# TODO: @rloganiv - Duping misconceptions for each positive variation is a lazy workaround to deal
# with the one-to-many relationship between misconceptions and positive variations; it works as
# intended and doesn't break anything, but involves unneccessary/convoluted data duplication and
# omission. Future solutions should be better about this...
@dataclass(eq=True, frozen=True)
class MisconceptionDataset:
    misconceptions: Tuple[Misconception]
    _sentences: Tuple[str]
    uid: str

    def __len__(self):
        return len(self.misconceptions)

    def __getitem__(self, index):
        return self.misconceptions[index]

    def __hash__(self):
        return self.uid

    @property
    def sentences(self):
        return list(self._sentences)

    @classmethod
    def from_jsonl(cls, f: StringIO):
        misconceptions = []
        sentences = []
        md5_hash = md5()
        for line in f:
            md5_hash.update(line.encode('utf-8'))
            obj = json.loads(line)
            immutable_obj = {k: _tuplify(v) for k, v in obj.items()}
            misconception = Misconception(**immutable_obj)
            for sentence in misconception.pos_variations:
                misconceptions.append(misconception)
                sentences.append(sentence)
        misconceptions = tuple(misconceptions)
        sentences = tuple(sentences)
        digest = md5_hash.digest()
        uid = int.from_bytes(digest, byteorder='big')
        return cls(misconceptions, sentences, uid)
