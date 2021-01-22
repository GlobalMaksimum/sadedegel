from typing import List
import numpy as np  # type: ignore
from ._base import ExtractiveSummarizer
from ..bblock import Sentences


class BM25Summarizer(ExtractiveSummarizer):
    """
    Assign a higher importance score based on BM25 score of the sentences within the document.

    k1 : Coefficient to be used for BM25 computation.
    b : Coefficient to be used for BM25 computation.

    normalize : bool, optional (default=True)
        If ``False``, return a raw score vector.
        Otherwise, return L2 normalized score vector.
    """
    tags = ExtractiveSummarizer.tags + ['self-supervised', 'ml', 'info-retrieval']

    def __init__(self, k1=1.25, b=0.75, normalize=True):
        super().__init__(normalize)

        self.k1 = k1
        self.b = b

    def _predict(self, sentences: List[Sentences]):
        return np.array([sent.bm25(k1=self.k1, b=self.b) for sent in sentences])
