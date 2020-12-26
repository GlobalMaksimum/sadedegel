from typing import List
import numpy as np

from ..tokenize import Sentences
from ._base import ExtractiveSummarizer


class TFIDFSummarizer(ExtractiveSummarizer):
    tags = ExtractiveSummarizer.tags + ['ml', 'tfidf']

    def __init__(self, normalize=True):
        super().__init__(normalize)

    def _predict(self, sents: List[Sentences]):
        return np.array([sent.tfidf().sum() for sent in sents])
