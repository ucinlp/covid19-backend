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
    category: str
    pos_variations: Tuple[str]
    neg_variations: Tuple[str]


@dataclass(eq=True, frozen=True)
class MisconceptionDataset:
    misconceptions: Tuple[Misconception]
    uid: str

    def __len__(self):
        return len(self.misconceptions)

    def __getitem__(self, index):
        return self.misconceptions[index]

    def __hash__(self):
        return self.uid

    @classmethod
    def from_jsonl(cls, f: StringIO):
        misconceptions = []
        md5_hash = md5()
        for line in f:
            md5_hash.update(line.encode('utf-8'))
            obj = json.loads(line)
            immutable_obj = {k: _tuplify(v) for k, v in obj.items()}
            misconception = Misconception(**immutable_obj)
            misconceptions.append(misconception)
        misconceptions = tuple(misconceptions)
        digest = md5_hash.digest()
        uid = int.from_bytes(digest, byteorder='big')
        return cls(misconceptions, uid)

    @property
    def sentences(self):
        return [m.canonical_sentence for m in self.misconceptions]
