from typing import List
import numpy as np

from ..tokenize import Sentences
from ._base import ExtractiveSummarizer


class TFIDFSummarizer(ExtractiveSummarizer):
    tags = ExtractiveSummarizer.tags + ['ml', 'tfidf']

    def __init__(self, tf_method=None, idf_method=None, drop_stopwords=False, drop_suffix=False,
                 drop_punct=False, lowercase=False, normalize=True, config=None):
        super().__init__(normalize)
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

        return np.array([sent.get_tfidf(tf_method=self.tf_method,
                                        idf_method=self.idf_method,
                                        drop_stopwords=self.drop_stopwords,
                                        drop_suffix=self.drop_suffix,
                                        drop_punct=self.drop_punct).sum() for sent in sents])
