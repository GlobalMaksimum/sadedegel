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
    tags = ExtractiveSummarizer.tags + ['self-supervised', 'ml', 'info-retrieval', 'bm']

    def __init__(self, k1=1.25, b=0.75,
                 tf_method=None, idf_method=None,
                 drop_stopwords=False, drop_suffix=False,
                 drop_punct=False, lowercase=False,
                 config=None, normalize=True):
        super().__init__(normalize)

        self.k1 = k1
        self.b = b

        self.drop_stopwords = drop_stopwords
        self.drop_suffix = drop_suffix
        self.drop_punct = drop_punct
        self.lowercase = lowercase

        self.config = config
        self.tf_method = tf_method
        self.idf_method = idf_method

    def _predict(self, sents: List[Sentences]):
        if self.config is None:
            self.config = sents[0].document.builder.config
        if self.tf_method is None:
            self.tf_method = self.config['tf']['method']
        if self.idf_method is None:
            self.idf_method = self.config['idf']['method']

        return np.array([sent.get_bm25(tf_method=self.tf_method,
                                       idf_method=self.idf_method,
                                       drop_stopwords=self.drop_stopwords,
                                       drop_suffix=self.drop_suffix,
                                       drop_punct=self.drop_punct,
                                       k1=self.k1, b=self.b).sum() for sent in sents])
