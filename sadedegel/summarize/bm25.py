from typing import List
import numpy as np  # type: ignore
from ._base import ExtractiveSummarizer
from ..bblock import Sentences


class BM25Summarizer(ExtractiveSummarizer):
    """
    Assign a higher importance score based on BM25 score of the sentences within the document.

    k1 : Coefficient to be used for BM25 computation.
    b : Coefficient to be used for BM25 computation.
    delta: BM25+ term

    normalize : bool, optional (default=True)
        If ``False``, return a raw score vector.
        Otherwise, return L2 normalized score vector.
    """
    tags = ExtractiveSummarizer.tags + ['self-supervised', 'ml', 'ir', 'bm']

    def __init__(self, tf_method, idf_method, k1=1.25, b=0.75, delta=0,
                 drop_stopwords=False, drop_suffix=False,
                 drop_punct=False, lowercase=False, normalize=True):
        super().__init__(normalize)

        self.k1 = k1
        self.b = b
        self.delta = delta

        self.drop_stopwords = drop_stopwords
        self.drop_suffix = drop_suffix
        self.drop_punct = drop_punct
        self.lowercase = lowercase

        self.tf_method = tf_method
        self.idf_method = idf_method

    def _predict(self, sents: List[Sentences]):
        return np.array([sent.get_bm25(tf_method=self.tf_method,
                                       idf_method=self.idf_method,
                                       drop_stopwords=self.drop_stopwords,
                                       drop_suffix=self.drop_suffix,
                                       drop_punct=self.drop_punct,
                                       k1=self.k1, b=self.b, delta=self.delta).sum() for sent in sents])
