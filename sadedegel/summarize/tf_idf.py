from typing import List
import numpy as np

from ..tokenize import Sentences
from ._base import ExtractiveSummarizer

from ..config import load_config


class TFIDFSummarizer(ExtractiveSummarizer):
    tags = ExtractiveSummarizer.tags + ['ml', 'tfidf', 'ir']

    def __init__(self, tf_method=None, idf_method=None, drop_stopwords=False, drop_suffix=False, drop_punct=False,
                 lowercase=False, normalize=True, kwargs={}):
        super().__init__(normalize)

        cfg = load_config()

        self.drop_stopwords = drop_stopwords
        self.drop_suffix = drop_suffix
        self.drop_punct = drop_punct
        self.lowercase = lowercase

        self.tf_method = tf_method if tf_method is not None else cfg['tf']['method']
        self.idf_method = idf_method if idf_method is not None else cfg['idf']['method']

        self.kwargs = kwargs

    def _predict(self, sents: List[Sentences]):
        return np.array([sent.get_tfidf(tf_method=self.tf_method,
                                        idf_method=self.idf_method,
                                        drop_stopwords=self.drop_stopwords,
                                        drop_suffix=self.drop_suffix,
                                        drop_punct=self.drop_punct, **self.kwargs).sum() for sent in sents])
