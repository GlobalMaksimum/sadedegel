from itertools import tee

import numpy as np
from rich.progress import track
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from ..bblock.doc import DocBuilder, Document


def check_type_all(X, expected_type=str):
    if not all(isinstance(x, expected_type) for x in X):
        raise ValueError(f"X should be an iterable {expected_type}. {type(X)} found")


def check_type(v, expected_type, error_msg: str) -> None:
    """Checks and compares of variable types of given variables.

    Parameters
    ----------
    v: Any
        Variable to be checked.
    expected_type: Any
        Expected type of variable.
    error_msg: str
        Error message.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the given and expected variables don't match.
    """
    if not isinstance(v, expected_type):
        raise ValueError(error_msg)


class OnlinePipeline(Pipeline):
    def partial_fit(self, X, y=None, **kwargs):
        """Implements minibatch type of training for given estimator.

        Parameters
        ----------
        X: array-like
            {array-like, sparse matrix} of shape (n_shape, n_features).
        y: array-like, default=None
            array-like of shape (n_samples,).
        kwargs: type
            Remaining arguments for current estimator.

        Returns
        -------
        self: object
            Fitted estimator.

        """
        for i, step in enumerate(self.steps):
            name, est = step
            est.partial_fit(X, y, **kwargs)
            if i < len(self.steps) - 1:
                X = est.transform(X)
        return self


class Text2Doc(BaseEstimator, TransformerMixin):
    """Creates text2doc converter for given tokenizer.

    Parameters
    ----------
    tokenizer: str, default='icu'
        Which tokenizer to be used.
    hashtag: bool, default=False
        Whether to tokenize hashtags alone or with related tokens.
    mention: bool, default=False
        Whether to tokenize mentions alone or with related tokens.
    emoji: bool, default=False
        Whether to tokenize emojis by symbols or all together.
    emoticon: bool, default=False
        Whether to tokenize emoticons by symbols or all together.
    progress_tracking: bool, default=True
    """

    Doc = None

    def __init__(self, tokenizer="icu", hashtag=False, mention=False, emoji=False, emoticon=False,
                 progress_tracking=True):
        self.tokenizer = tokenizer
        self.hashtag = hashtag
        self.mention = mention
        self.emoji = emoji
        self.emoticon = emoticon
        self.progress_tracking = progress_tracking
        # TODO: Add sadedegel version

        self.init()

    def init(self):
        if Text2Doc.Doc is None:
            if hasattr(self, 'hashtag') and hasattr(self, 'mention') and hasattr(self, 'emoji') and hasattr(
                    self, 'emoticon'):
                Text2Doc.Doc = DocBuilder(tokenizer=self.tokenizer, tokenizer__hashtag=self.hashtag,
                                          tokenizer__mention=self.mention, tokenizer__emoji=self.emoji,
                                          tokenizer__emoticon=self.emoticon)
            else:
                Text2Doc.Doc = DocBuilder(tokenizer=self.tokenizer, tokenizer__hashtag=False,
                                          tokenizer__mention=False, tokenizer__emoji=False,
                                          tokenizer__emoticon=False)

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None):
        """Transforms given list of strings into list of sadedegel's Doc objects which can be tokenized.

        Parameters
        ----------
        X: array-like
            List of strings to be transformed.

        Returns
        -------
        docs: array-like
            List of sadedegel.bblock.doc.Document objects.

        Raises
        ------
        ValueError
            If the X contains no valid documents.
        """
        if isinstance(X, list):
            check_type_all(X)
            n_total = len(X)
        else:
            X1, X2, X = tee(X, 3)

            check_type_all(X1)
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
    """Coverts a collection of text documents to a matrix of occurrences.

    Parameters
    ----------
    n_features: int, default=1048576
        The number of features(columns) in output matrices.
        Small numbers can cause hash collisions.
    prefix_range: tuple (min_n, max_n), default=(3,5)
        The lower and upper boundary of the range of n-values for prefixes (Characters).
    alternate_sign: bool, default=True
        When True, an alternating sign is added to features as to
        approximately converse the linear product in the hashed space.
    """
    def __init__(self, n_features: int = 1048576, prefix_range: tuple = (3, 5), *, alternate_sign: bool = True):
        check_type(prefix_range, tuple, f"prefix_range should be of tuple type. {type(prefix_range)} found.")
        self.n_features = n_features
        self.alternate_sign = alternate_sign

        self.prefix_range = prefix_range

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None, **kwargs):
        return self

    def transform(self, docs):
        """Takes sadedegel doc objects and returns matrix of hashed features.

        Parameters
        ----------
        docs: sadedegel.bblock.doc.Document or list
            List of sadedegel.bblock.doc.Document objects or single Doc object.
        Returns
        -------
        csr_matrix: array-like
            scipy.sparse.csr of shape (n_samples, n_features)
        """
        def feature_iter():
            if hasattr(self, 'prefix_range'):
                for d in docs:
                    yield [(f'prefix{p_ix}', t.lower_[:p_ix]) for p_ix in
                           range(self.prefix_range[0], self.prefix_range[1] + 1) for t in d.tokens]
            else:
                for d in docs:
                    yield [('prefix5', t.lower_[:5]) for t in d.tokens] + [('prefix3', t.lower_[:3]) for t in
                                                                           d.tokens]

        return FeatureHasher(self.n_features, alternate_sign=self.alternate_sign, input_type="pair",
                             dtype=np.float32).transform(feature_iter())


class TfidfVectorizer(SadedegelVectorizer):
    """Transforms a count matrix to a normalized tf or tf-idf representation.
    Parameters
    ----------
    tf_method: str, default='raw'
        Type of term frequency method. Can be one of:
        ['raw', 'binary', 'freq', 'log_norm', 'double_norm']
    idf_method: str, default='probabilistic'
        Type of inverse document frequency method. Can be one of:
        ['smooth', 'probabilistic', 'unary']
    drop_stopwords: bool, default=True
        Whether to drop or keep stopwords.
    lowercase: bool, default=True
        Whether to lowercase or keep original cases of given text.
    drop_suffix: bool, default=True
        Whether to drop suffixes or keep original version of the text.
    drop_punct: bool, default=True
        Whether to drop or keep punctuations.
    show_progress: bool, default=True
        Whether to keep track of progress or not.
    """
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
        """Takes list of sadedegel documents and returns matrix of tf-idf features which is pretrained.

        Parameters
        ----------
        X: array-like
            List of sadedegel.bblock.doc.Document objects.
        Returns
        -------
        csr_matrix: array-like
            scipy.sparse.csr of shape (n_samples, n_features)

        Raises
        ------
        ValueError
            If the X contains no valid documents.
        """
        if isinstance(X, list):
            check_type_all(X, Document)
            n_total = len(X)
        else:
            X1, X2, X = tee(X, 3)

            check_type_all(X1, Document)
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
    """sadedegel's 'Best Match 25' implementation.

    Parameters
    ----------
    tf_method: str, default='raw'
        Type of term frequency method. Can be one of:
        ['raw', 'binary', 'freq', 'log_norm', 'double_norm']
    idf_method: str, default='probabilistic'
        Type of inverse document frequency method. Can be one of:
        ['smooth', 'probabilistic', 'unary']
    k1: float, default=1.25
        Determines the term frequency saturation.
    b: float, default=0.75
        Ratio of the document length to be multiplied.
    delta: float, default=0
        A constant value for lower bounding.
    drop_stopwords: bool, default=True
        Whether to drop or keep stopwords.
    lowercase: bool, default=True
        Whether to lowercase or keep original cases of given text.
    drop_suffix: bool, default=True
        Whether to drop suffixes or keep original version of the text.
    drop_punct: bool, default=True
        Whether to drop or keep punctuations.
    show_progress: bool, default=True
        Whether to keep track of progress or not.
    """
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
        """Takes list of sadedegel documents and returns matrix of features which is pretrained.

        Parameters
        ----------
        X: array-like
            List of sadedegel.bblock.doc.Document objects.
        Returns
        -------
        csr_matrix: array-like
            scipy.sparse.csr of shape (n_samples, n_features)
        Raises
        ------
        ValueError
            If the X contains no valid documents.
        """
        if isinstance(X, list):
            check_type_all(X, Document)
            n_total = len(X)
        else:
            X1, X2, X = tee(X, 3)

            check_type_all(X1, Document)
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


class PreTrainedVectorizer(BaseEstimator, TransformerMixin):
    """Pretrained vectorizer using transformer based embeddings.

    Parameters
    ----------
    model: str
        Name of the huggingface pretrained model.
    do_sents: bool, default=False
        Whether to get sentence or raw document embeddings.
    show_progress: bool, default=True
        Whether to keep track of progress or not.
    """
    Doc = None

    def __init__(self, model: str, do_sents = False, progress_tracking = True):

        super().__init__()
        self.model = model
        self.do_sents = do_sents
        self.progress_tracking = progress_tracking

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Takes list of sadedegel documents consists of sentences and returns matrix of features which is pretrained.

        Parameters
        ----------
        X: array-like
            List of sadedegel.bblock.doc.Document objects.
        Returns
        -------
        csr_matrix: array-like
            scipy.sparse.csr
        """
        if PreTrainedVectorizer.Doc is None:
            PreTrainedVectorizer.Doc = DocBuilder()

        vecs = []
        n_total = 0
        for text in tqdm(X, disable=not hasattr(self, 'progress_tracking') or not self.progress_tracking, unit="doc"):
            d = PreTrainedVectorizer.Doc(text)
            vecs.append(d.get_pretrained_embedding(architecture=self.model, do_sents=self.do_sents))
            if self.do_sents:
                n_total += len(d)
            else:
                n_total += 1
        vector_shape = vecs[0].shape[1]

        return csr_matrix(np.vstack(vecs), dtype=np.float32, shape=(n_total, vector_shape))
