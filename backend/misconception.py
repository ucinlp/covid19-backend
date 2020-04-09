"""
Objects that load/store/validate misconception data.
"""
from dataclasses import dataclass
from io import StringIO
import json
from typing import Tuple


# Define misconceptions using dataclasses so loaded data is automatically validated and hashable
@dataclass(eq=True, frozen=True)
class Misconception:
    sentence: str
    link: str


@dataclass(eq=True, frozen=True)
class MisconceptionDataset:
    misconceptions: Tuple[Misconception]

    def __len__(self):
        return len(self.misconceptions)

    def __getitem__(self, index):
        return self.misconceptions[index]

    @classmethod
    def from_jsonl(cls, f: StringIO):
        misconceptions = []
        for line in f:
            obj = json.loads(line)
            misconception = Misconception(**obj)
            misconceptions.append(misconception)
        misconceptions = tuple(misconceptions)
        return cls(misconceptions)

    @property
    def sentences(self):
        return [m.sentence for m in self.misconceptions]

    @property
    def links(self):
        return [m.link for m in self.misconceptions]
