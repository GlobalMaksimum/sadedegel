from itertools import tee

import numpy as np
from rich.progress import track
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import FeatureHasher

from ..bblock.doc import DocBuilder, Document
from ..bblock.token import Token
from tqdm import tqdm


def check_type(X):
    if not all(isinstance(x, str) for x in X):
        raise ValueError(f"X should be an iterable string. {type(X)} found")


def check_doc_type(X):
    if not all(isinstance(x, Document) for x in X):
        raise ValueError(f"X should be an iterable sadedegel Document. {type(X)} found")


class OnlinePipeline(Pipeline):
    def partial_fit(self, X, y=None, **kwargs):
        for i, step in enumerate(self.steps):
            name, est = step
            est.partial_fit(X, y, **kwargs)
            if i < len(self.steps) - 1:
                X = est.transform(X)
        return self


class Text2Doc(BaseEstimator, TransformerMixin):
    Doc = None

    def __init__(self, tokenizer="icu", hashtag=False, mention=False, emoji=False, progress_tracking=True):
        self.tokenizer = tokenizer
        self.hashtag = hashtag
        self.mention = mention
        self.emoji = emoji
        self.progress_tracking = progress_tracking
        # TODO: Add sadedegel version

        self.init()

    def init(self):
        if Text2Doc.Doc is None:
            Text2Doc.Doc = DocBuilder(tokenizer=self.tokenizer, tokenizer__hashtag=self.hashtag,
                                      tokenizer__mention=self.mention, tokenizer__emoji=self.emoji)

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None, **kwargs):
        return self

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

        docs = []

        for text in tqdm(X, disable=not hasattr(self, 'progress_tracking') or not self.progress_tracking, unit="doc"):
            docs.append(Text2Doc.Doc(text))

        return docs


class SadedegelVectorizer(BaseEstimator, TransformerMixin):
    """Sadedegel feature extraction TransformerMixin s don't require fit calls."""

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None, **kwargs):
        return self


class HashVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=1048576, *, alternate_sign=True):
        self.n_features = n_features
        self.alternate_sign = alternate_sign

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None, **kwargs):
        return self

    def transform(self, docs):
        def feature_iter():
            for d in docs:
                yield [('prefix5', t.lower_[:5]) for t in d.tokens] + [('prefix3', t.lower_[:3]) for t in
                                                                       d.tokens]

        return FeatureHasher(self.n_features, alternate_sign=self.alternate_sign, input_type="pair",
                             dtype=np.float32).transform(feature_iter())


class TfidfVectorizer(SadedegelVectorizer):
    def __init__(self, *, tf_method='raw', idf_method='probabilistic', drop_stopwords=True,
                 lowercase=True,
                 drop_suffix=True, drop_punct=True, show_progress=True):
        super().__init__()

        self.tf_method = tf_method
        self.idf_method = idf_method
        self.lowercase = lowercase
        self.drop_suffix = drop_suffix
        self.drop_stopwords = drop_stopwords
        self.drop_punct = drop_punct
        self.show_progress = show_progress

    def transform(self, X, y=None):
        if isinstance(X, list):
            check_doc_type(X)
            n_total = len(X)
        else:
            X1, X2, X = tee(X, 3)

            check_doc_type(X1)
            n_total = sum((1 for _ in X2))

        if n_total == 0:
            raise ValueError(f"Ensure that X contains at least one valid document. Found {n_total}")

        indptr = [0]
        indices = []
        data = []
        for doc in track(X, total=n_total, description="Transforming document(s)", update_period=1,
                         disable=not self.show_progress):
            if self.lowercase:
                n_vocabulary = doc.builder.tokenizer.vocabulary.size
            else:
                n_vocabulary = doc.builder.tokenizer.vocabulary.size_cs

            tfidf = doc.get_tfidf(self.tf_method, self.idf_method, drop_stopwords=self.drop_stopwords,
                                  lowercase=self.lowercase,
                                  drop_suffix=self.drop_suffix,
                                  drop_punct=self.drop_punct)

            for idx in tfidf.nonzero()[0]:
                indices.append(idx)
                data.append(tfidf[idx])

            indptr.append(len(indices))

        return csr_matrix((data, indices, indptr), dtype=np.float32, shape=(n_total, n_vocabulary))


class BM25Vectorizer(SadedegelVectorizer):
    def __init__(self, *, tf_method='raw', idf_method='probabilistic', k1=1.25, b=0.75, delta=0,
                 drop_stopwords=True,
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

    def transform(self, X, y=None):
        if isinstance(X, list):
            check_doc_type(X)
            n_total = len(X)
        else:
            X1, X2, X = tee(X, 3)

            check_doc_type(X1)
            n_total = sum((1 for _ in X2))

        if n_total == 0:
            raise ValueError(f"Ensure that X contains at least one valid document. Found {n_total}")

        indptr = [0]
        indices = []
        data = []
        for doc in track(X, total=n_total, description="Transforming document(s)", update_period=1,
                         disable=not self.show_progress):

            if self.lowercase:
                n_vocabulary = doc.builder.tokenizer.vocabulary.size
            else:
                n_vocabulary = doc.builder.tokenizer.vocabulary.size_cs

            bm25 = doc.get_bm25(self.tf_method, self.idf_method, drop_stopwords=self.drop_stopwords,
                                lowercase=self.lowercase,
                                drop_suffix=self.drop_suffix,
                                drop_punct=self.drop_punct,
                                k1=self.k1, b=self.b, delta=self.delta)

            for idx in bm25.nonzero()[0]:
                indices.append(idx)
                data.append(bm25[idx])

            indptr.append(len(indices))

        return csr_matrix((data, indices, indptr), dtype=np.float32, shape=(n_total, n_vocabulary))
