from itertools import tee

from rich.progress import track
from sklearn.base import BaseEstimator, TransformerMixin

from scipy.sparse import csr_matrix

import numpy as np

from ..config import config_context


def check_type(X):
    if not all(isinstance(x, str) for x in X):
        raise ValueError(f"X should be an iterable string (documents)")


class TfidfVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, *, tf_method='raw', idf_method='probabilistic', drop_stopwords=True, lowercase=True,
                 drop_suffix=True, drop_punct=True):
        self.tf_method = tf_method
        self.idf_method = idf_method
        self.lowercase = lowercase
        self.drop_suffix = drop_suffix
        self.drop_stopwords = drop_stopwords
        self.drop_punct = drop_punct

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, list):
            check_type(X)
            n_total = len(X)
        else:
            X1, X2, X = tee(X, 3)

            check_type(X1)
            n_total = sum((1 for _ in X2))

        with config_context(tokenizer="bert") as Doc:
            indptr = [0]
            indices = []
            data = []
            for doc in track(X, total=n_total, description="Transforming corpus", update_period=1):
                d = Doc(doc)
                n_vocabulary = len(d.builder.tokenizer.vocabulary)
                tfidf = d.get_tfidf(self.tf_method, self.idf_method, drop_stopwords=self.drop_stopwords,
                                    lowercase=self.lowercase,
                                    drop_suffix=self.drop_suffix,
                                    drop_punct=self.drop_punct)

                for idx in tfidf.nonzero()[0]:
                    indices.append(idx)
                    data.append(tfidf[idx])

                indptr.append(len(indices))

            return csr_matrix((data, indices, indptr), dtype=np.float32, shape=(n_total, n_vocabulary))
