from collections import defaultdict
from typing import List
from math import log
import numpy as np

from ..tokenize import Doc, Sentences
from ._base import ExtractiveSummarizer


class TFIDFSummarizer(ExtractiveSummarizer):
    tags = ExtractiveSummarizer.tags + ['ml']

    def __init__(self, normalize=True):
        super().__init__(normalize)

    def _predict(self, sents: List[Sentences]):
        return np.array([sent.tfidf().sum() for sent in sents])
