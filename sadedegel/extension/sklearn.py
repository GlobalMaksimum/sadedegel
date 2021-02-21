from itertools import tee

import numpy as np
from rich.progress import track
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from ..config import config_context


def check_type(X):
    if not all(isinstance(x, str) for x in X):
        raise ValueError(f"X should be an iterable string (documents)")


class OnlinePipeline(Pipeline):
    def partial_fit(self, X, y=None, **kwargs):
        for i, step in enumerate(self.steps):
            name, est = step
            est.partial_fit(X, y, **kwargs)
            if i < len(self.steps) - 1:
                X = est.transform(X)
        return self


class SadedegelVectorizer(BaseEstimator, TransformerMixin):
    """Sadedegel feature extraction TransformerMixin s don't require fit calls."""

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None, **kwargs):
        return self


class TfidfVectorizer(SadedegelVectorizer):
    def __init__(self, *, tf_method='raw', idf_method='probabilistic', drop_stopwords=True, lowercase=True,
                 drop_suffix=True, drop_punct=True, show_progress=True):
        super().__init__()

        self.tf_method = tf_method
        self.idf_method = idf_method
        self.lowercase = lowercase
        self.drop_suffix = drop_suffix
        self.drop_stopwords = drop_stopwords
        self.drop_punct = drop_punct
        self.show_progress = show_progress

        self.Doc = None

    def transform(self, X, y=None):
        if isinstance(X, list):
            check_type(X)
            n_total = len(X)
        else:
            X1, X2, X = tee(X, 3)

            check_type(X1)
            n_total = sum((1 for _ in X2))

        if n_total == 0:
            raise ValueError(f"Ensure that X contains at least one valid document. Found {n_total}")

        if self.Doc is None:
            with config_context(tokenizer="bert") as Doc:
                self.Doc = Doc

        indptr = [0]
        indices = []
        data = []
        for doc in track(X, total=n_total, description="Transforming document(s)", update_period=1,
                         disable=not self.show_progress):
            d = self.Doc(doc)
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


class BM25Vectorizer(SadedegelVectorizer):
    def __init__(self, *, tf_method='raw', idf_method='probabilistic', k1=1.25, b=0.75, delta=0, drop_stopwords=True,
                 lowercase=True, drop_suffix=True, drop_punct=True, show_progress=True):

        super().__init__()

        self.tf_method = tf_method
        self.idf_method = idf_method
        self.lowercase = lowercase
        self.drop_suffix = drop_suffix
        self.drop_stopwords = drop_stopwords
        self.drop_punct = drop_punct
        self.show_progress = show_progress
        self.k1 = k1
        self.b = b
        self.delta = delta

        self.Doc = None

    def transform(self, X, y=None):
        if isinstance(X, list):
            check_type(X)
            n_total = len(X)
        else:
            X1, X2, X = tee(X, 3)

            check_type(X1)
            n_total = sum((1 for _ in X2))

        if n_total == 0:
            raise ValueError(f"Ensure that X contains at least one valid document. Found {n_total}")

        if self.Doc is None:
            with config_context(tokenizer="bert") as Doc:
                self.Doc = Doc

        indptr = [0]
        indices = []
        data = []
        for doc in track(X, total=n_total, description="Transforming document(s)", update_period=1,
                         disable=not self.show_progress):
            d = self.Doc(doc)
            n_vocabulary = len(d.builder.tokenizer.vocabulary)
            bm25 = d.get_bm25(self.tf_method, self.idf_method, drop_stopwords=self.drop_stopwords,
                              lowercase=self.lowercase,
                              drop_suffix=self.drop_suffix,
                              drop_punct=self.drop_punct,
                              k1=self.k1, b=self.b, delta=self.delta)

            for idx in bm25.nonzero()[0]:
                indices.append(idx)
                data.append(bm25[idx])

            indptr.append(len(indices))

        return csr_matrix((data, indices, indptr), dtype=np.float32, shape=(n_total, n_vocabulary))
