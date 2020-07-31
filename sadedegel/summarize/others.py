from typing import List
import numpy as np

from ..tokenize.helper import Sentences
from ._base import ExtractiveSummarizer


class CompositeSummarizer(ExtractiveSummarizer):
    def predict(self, sents: List[Sentences]) -> np.ndarray:
        pass
