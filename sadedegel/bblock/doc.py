import re
import sys
from collections import Counter
from functools import partial
from typing import List

import numpy as np  # type:ignore
from rich.console import Console
from scipy.sparse import csr_matrix
from cached_property import cached_property

import warnings

warnings.filterwarnings("ignore")

from .token import Token, IDF_METHOD_VALUES, IDFImpl
from .util import tr_lower, __tr_lower_abbrv__, flatten, pad, normalize_tokenizer_name, __transformer_model_mapper__, \
    ArchitectureNotFound, TransformerModel
from .word_tokenizer import WordTokenizer
from ..config import load_config
from ..metrics import rouge1_score
from ..ml.sbd import load_model

console = Console()


class Span:
    """Span class to store raw string as the smallest unit within sadedegel object hiearchy.
    Attributes of this class store rule based user-defined features of a span for training a sentence boundary detection model.

    """
    def __init__(self, i: int, span, doc):
        self.doc = doc
        self.i = i
        self.value = span

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text

    @property
    def text(self):
        """Raw string that constitute the Span instance.

        Returns
        -------
        raw: str
        """
        return self.doc.raw[slice(*self.value)]

    def span_features(self):
        """Features of the span.

        Returns
        -------
        features: dict
        """
        is_first_span = self.i == 0
        is_last_span = self.i == len(self.doc._spans) - 1
        word = self.doc.raw[slice(*self.value)]

        if not is_first_span:
            word_m1 = self.doc.raw[slice(*self.doc._spans[self.i - 1].value)]
        else:
            word_m1 = '<D>'

        if not is_last_span:
            word_p1 = self.doc.raw[slice(*self.doc._spans[self.i + 1].value)]
        else:
            word_p1 = '</D>'

        features = {}

        # All upper features
        if word_m1.isupper() and not is_first_span:
            features["PREV_ALL_CAPS"] = True

        if word.isupper():
            features["ALL_CAPS"] = True

        if word_p1.isupper() and not is_last_span:
            features["NEXT_ALL_CAPS"] = True

        # All lower features
        if word_m1.islower() and not is_first_span:
            features["PREV_ALL_LOWER"] = True

        if word.islower():
            features["ALL_LOWER"] = True

        if word_p1.islower() and not is_last_span:
            features["NEXT_ALL_LOWER"] = True

        # Century
        if tr_lower(word_p1).startswith("yüzyıl") and not is_last_span:
            features["NEXT_WORD_CENTURY"] = True

        # Number
        if (tr_lower(word_p1).startswith("milyar") or tr_lower(word_p1).startswith("milyon") or tr_lower(
                word_p1).startswith("bin")) and not is_last_span:
            features["NEXT_WORD_NUMBER"] = True

        # Percentage
        if tr_lower(word_m1).startswith("yüzde") and not is_first_span:
            features["PREV_WORD_PERCENTAGE"] = True

        # In parenthesis feature
        if word[0] == '(' and word[-1] == ')':
            features["IN_PARENTHESIS"] = True
        # this is test comment
        # Suffix features
        m = re.search(r'\W+$', word)

        if m:
            subspan = m.span()
            suff2 = word[slice(*subspan)][:2]

            if all((c in '!\').?’”":;…') for c in suff2):
                features['NON-ALNUM SUFFIX2'] = suff2

        if word_p1[0] in '-*◊' and not is_last_span:
            features["NEXT_BULLETIN"] = True

        # title features
        # if word_m1.istitle() and not is_first_span:
        #     features["PREV_TITLE"] = 1

        # NOTE: 'Beşiktaş’ın' is not title by python :/
        m = re.search(r'\w+', word)

        if m:
            subspan = m.span()
            if word[slice(*subspan)].istitle():
                features["TITLE"] = True

        m = re.search(r'\w+', word_p1)

        if m:
            subspan = m.span()

            if word_p1[slice(*subspan)].istitle() and not is_last_span:
                features["NEXT_TITLE"] = True

        # suffix features
        for name, symbol in zip(['DOT', 'ELLIPSES', 'ELLIPSES', 'EXCLAMATION', 'QUESTION'],
                                ['.', '...', '…', '?', '!']):
            if word.endswith(symbol):
                features[f"ENDS_WITH_{name}"] = True

        # prefix abbreviation features
        if '.' in word:

            if any(tr_lower(word).startswith(abbrv) for abbrv in __tr_lower_abbrv__):
                features["PREFIX_IS_ABBRV"] = True
            if any(tr_lower(word).endswith(abbrv) for abbrv in __tr_lower_abbrv__):
                features["SUFFIX_IS_ABBRV"] = True
            else:
                prefix = word.split('.', maxsplit=1)[0]
                features["PREFIX_IS_DIGIT"] = prefix.isdigit()

        # if '.' in word_m1:
        #     prefix_m1 = word_m1.split('.', maxsplit=1)[0]
        #
        #     if tr_lower(prefix_m1) in __tr_lower_abbrv__ and not is_first_span:
        #         features[f"PREV_PREFIX_IS_ABBRV"] = True
        #     else:
        #         features["PREV_PREFIX_IS_DIGIT"] = prefix_m1.isdigit()
        #
        # if '.' in word_p1:
        #     prefix_p1 = word_p1.split('.', maxsplit=1)[0]
        #
        #     if tr_lower(prefix_p1) in __tr_lower_abbrv__ and not is_last_span:
        #         features[f"NEXT_PREFIX_IS_ABBRV"] = True
        #     else:
        #         features["NEXT_PREFIX_IS_DIGIT"] = prefix_p1.isdigit()

        return features


TF_BINARY, TF_RAW, TF_FREQ, TF_LOG_NORM, TF_DOUBLE_NORM = "binary", "raw", "freq", "log_norm", "double_norm"
TF_METHOD_VALUES = [TF_BINARY, TF_RAW, TF_FREQ, TF_LOG_NORM, TF_DOUBLE_NORM]


class TFImpl:
    """Base implementation of Term Frequency. Includes various TF calculation methods for inheritance of sadedegel.bblock.Sentences and sadedegel.bblock.Document.

    ...
    Methods
    -------
    raw_tf: numpy.ndarray
    binary_tf: numpy.ndarray
    freq_tf: numpy.ndarray
    log_norm_tf: numpy.ndarray
    double_norm_tf: numpy.ndarray
    get_tf: numpy.ndarray
    """
    def __init__(self):
        pass

    def raw_tf(self, drop_stopwords=False, lowercase=False, drop_suffix=False, drop_punct=False) -> np.ndarray:
        """Calculate TF with raw token counts.

        Parameters
        ----------
        drop_stopwords: bool
            Drop stopwords from the sequence.
        lowercase: bool
            Lowerize all tokens in sequence.
        drop_suffix: bool
            Drop suffixes from sequence that is tokenized by sadedegel.bblock.BertTokenizer.
        drop_punct: bool
            Drop punctuation from the sequence.

        Returns
        -------
        tf: numpy.ndarray
            Sparse TF vector representation

        """
        if lowercase:
            v = np.zeros(self.vocabulary.size)
        else:
            v = np.zeros(self.vocabulary.size_cs)

        if lowercase:
            tokens = [t.lower_ for t in self.tokens]
        else:
            tokens = [t.word for t in self.tokens]

        counter = Counter(tokens)

        for token in tokens:
            t = Token(token)
            if t.is_oov or (drop_stopwords and t.is_stopword) or (drop_suffix and t.is_suffix) or (
                    drop_punct and t.is_punct):
                continue

            if lowercase:
                v[t.id] = counter[token]
            else:
                v[t.id_cs] = counter[token]

        return v

    def binary_tf(self, drop_stopwords=False, lowercase=False, drop_prefix=False, drop_punct=False) -> np.ndarray:
        """Calculate TF with binary occurence. i.e. One-hot representation.

        Parameters
        ----------
        drop_stopwords: bool
            Drop stopwords from the sequence.
        lowercase: bool
            Lowerize all tokens in sequence.
        drop_suffix: bool
            Drop suffixes from sequence that is tokenized by sadedegel.bblock.BertTokenizer.
        drop_punct: bool
            Drop punctuation from the sequence.

        Returns
        -------
        tf: numpy.ndarray
            Sparse TF vector representation

        """
        return self.raw_tf(drop_stopwords, lowercase, drop_prefix, drop_punct).clip(max=1)

    def freq_tf(self, drop_stopwords=False, lowercase=False, drop_prefix=False, drop_punct=False) -> np.ndarray:
        """Calculate TF with normalized token counts i.e. frequency.

        Parameters
        ----------
        drop_stopwords: bool
            Drop stopwords from the sequence.
        lowercase: bool
            Lowerize all tokens in sequence.
        drop_suffix: bool
            Drop suffixes from sequence that is tokenized by sadedegel.bblock.BertTokenizer.
        drop_punct: bool
            Drop punctuation from the sequence.

        Returns
        -------
        tf: numpy.ndarray
            Sparse TF vector representation

        """
        tf = self.raw_tf(drop_stopwords, lowercase, drop_prefix, drop_punct)

        normalization = tf.sum()

        if normalization > 0:
            return tf / normalization
        else:
            return tf

    def log_norm_tf(self, drop_stopwords=False, lowercase=False, drop_prefix=False, drop_punct=False) -> np.ndarray:
        """Calculate TF with log normalized token counts.

        Parameters
        ----------
        drop_stopwords: bool
            Drop stopwords from the sequence.
        lowercase: bool
            Lowerize all tokens in sequence.
        drop_suffix: bool
            Drop suffixes from sequence that is tokenized by sadedegel.bblock.BertTokenizer.
        drop_punct: bool
            Drop punctuation from the sequence.

        Returns
        -------
        tf: numpy.ndarray
            Sparse TF vector representation

        """
        return np.log1p(self.raw_tf(drop_stopwords, lowercase, drop_prefix, drop_punct))

    def double_norm_tf(self, drop_stopwords=False, lowercase=False, drop_prefix=False, drop_punct=False,
                       k=0.5) -> np.ndarray:
        """Calculate TF with normalized token counts i.e. frequency.

        Parameters
        ----------
        drop_stopwords: bool
            Drop stopwords from the sequence.
        lowercase: bool
            Lowerize all tokens in sequence.
        drop_suffix: bool
            Drop suffixes from sequence that is tokenized by sadedegel.bblock.BertTokenizer.
        drop_punct: bool
            Drop punctuation from the sequence.
        k: float
            Weighting parameter.

        Returns
        -------
        tf: numpy.ndarray
            Sparse TF vector representation

        Raises
        ------
        ValueError
            If k is not in range (0, 1).
        """
        if not (0 < k < 1):
            raise ValueError(f"Ensure that 0 < k < 1 for double normalization term frequency calculation ({k} given)")

        tf = self.raw_tf(drop_stopwords, lowercase, drop_prefix, drop_punct)
        normalization = tf.max()

        if normalization > 0:
            return k + (1 - k) * (tf / normalization)
        else:
            return tf

    def get_tf(self, method=TF_BINARY, drop_stopwords=False, lowercase=False, drop_suffix=False, drop_punct=False,
               **kwargs) -> np.ndarray:
        if method == TF_BINARY:
            return self.binary_tf(drop_stopwords, lowercase, drop_suffix, drop_punct)
        elif method == TF_RAW:
            return self.raw_tf(drop_stopwords, lowercase, drop_suffix, drop_punct)
        elif method == TF_FREQ:
            return self.freq_tf(drop_stopwords, lowercase, drop_suffix, drop_punct)
        elif method == TF_LOG_NORM:
            return self.log_norm_tf(drop_stopwords, lowercase, drop_suffix, drop_punct)
        elif method == TF_DOUBLE_NORM:
            return self.double_norm_tf(drop_stopwords, lowercase, drop_suffix, drop_punct, **kwargs)
        else:
            raise ValueError(f"Unknown tf method ({method}). Choose one of {TF_METHOD_VALUES}")

    @property
    def tf(self):
        tf = self.config['tf']['method']
        drop_stopwords = self.config['default'].getboolean('drop_stopwords')
        lowercase = self.config['default'].getboolean('lowercase')
        drop_suffix = self.config['bert'].getboolean('drop_suffix')
        drop_punct = self.config['default'].getboolean('drop_punct')

        return self.get_tf(tf, drop_stopwords, lowercase, drop_suffix, drop_punct)


class BM25Impl:
    """Base Implementation for BM25 relevance score. Calculate BM25 ant other variants (BM11, BM15) based on user defined configuration parameters.
    For more information on BM25 implementation refer to: https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf

    ...
    Methods
    -------
    get_bm25: float
        Retrieve BM25 relevance score of the sentence w.r.t. its document.
    get_bm11: float
        Retrieve BM11 relevance score of the sentence w.r.t. its document.
    get_bm15: float
        Retrieve BM15 relevance score of the sentence w.r.t. its document.
    """
    def __init__(self):
        pass

    def get_bm25(self, tf_method: str, idf_method: str, k1: float, b: float, delta: float = 0, drop_stopwords=False,
                 lowercase=False,
                 drop_suffix=False,
                 drop_punct=False, **kwargs):
        """Retrieve BM25 relevance score of the sentence w.r.t. its document.

        Parameters
        ----------
        tf_method: str
            Term Frequency calculation method.
        idf_method: str
            Inverse Document Frequency calculation method.
        k1: int
            Smoothing term for weighting in set.
        b: int
            Weighting for sentence_len to document_len ratio.
        delta: float
            Normalization term for lower bounding de-weighting for very long documents.
        drop_stopwords: bool
            Drop stopwords from the sequence.
        lowercase: bool
            Lowerize all tokens in sequence.
        drop_suffix: bool
            Drop suffixes from sequence that is tokenized by sadedegel.bblock.BertTokenizer.
        drop_punct: bool
            Drop punctuation from the sequence.
        **kwargs: dict, optional

        Returns
        -------
        bm25: float
        """
        tf = self.get_tf(tf_method, drop_stopwords, lowercase, drop_suffix, drop_punct, **kwargs)
        idf = self.get_idf(idf_method, drop_stopwords, lowercase, drop_suffix, drop_punct, **kwargs)

        bm25 = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (len(self) / self.avgdl))) + delta)

        return bm25

    def get_bm11(self, tf_method, idf_method, k1, avgdl, delta=0, drop_stopwords=False, lowercase=False,
                 drop_suffix=False,
                 drop_punct=False, **kwargs):
        """Retrieve BM11 relevance score of the sentence w.r.t. its document.

        Parameters
        ----------
        **kwargs: dict, optional

        Returns
        -------
        bm11: float
        """
        return self.get_bm25(tf_method, idf_method, k1, 1, avgdl, delta, drop_stopwords, lowercase, drop_suffix,
                             drop_punct,
                             kwargs)

    def get_bm15(self, tf_method, idf_method, k1, avgdl, delta=0, drop_stopwords=False, lowercase=False,
                 drop_suffix=False,
                 drop_punct=False, **kwargs):
        """Retrieve BM15 relevance score of the sentence w.r.t. its document.

        Parameters
        ----------
        **kwargs: dict, optional

        Returns
        -------
        bm15: float
        """
        return self.get_bm25(tf_method, idf_method, k1, 0, avgdl, delta, drop_stopwords, lowercase, drop_suffix,
                             drop_punct,
                             kwargs)


class Sentences(TFImpl, IDFImpl, BM25Impl):
    """Sentences class is a sequence of Token objects. Access tokens that constitute the sentence.
    Generate BoW or PreTrainedTransformer model based embeddings.

    ...
    Attributes
    ----------
    id: int
        ID of the sentence. i.e. order in the Document that it is present.
    text: str
        Raw string of the sentence.
    document: sadedegel.Doc
        Document object that the Sentences instance is part of. Parent document node of the child sentence.
    config: dict
        Configuration that attribute calculations depend on.
    tf_method: str
        Term Frequency calculation method for term frequency of a token within a sentence.

    Methods
    -------
    get_tfidf
        Returns sparse tfidf representation of the sentence calculated over extended news corpus vocabulary dump.
    """

    def __init__(self, id_: int, text: str, doc, config: dict = {}):
        TFImpl.__init__(self)
        IDFImpl.__init__(self)
        BM25Impl.__init__(self)

        self.id = id_
        self.text = text

        self.document = doc
        self.config = doc.builder.config
        self._bert = None

        # No fail back to config read in here because this will slow down sentence instantiation extremely.
        self.tf_method = config['tf']['method']

        if self.tf_method == TF_BINARY:
            self.f_tf = self.binary_tf
        elif self.tf_method == TF_RAW:
            self.f_tf = self.raw_tf
        elif self.tf_method == TF_FREQ:
            self.f_tf = self.freq_tf
        elif self.tf_method == TF_LOG_NORM:
            self.f_tf = self.log_norm_tf
        elif self.tf_method == TF_DOUBLE_NORM:
            k = config.getfloat('tf', 'double_norm_k')

            if not 0 < k < 1:
                raise ValueError(
                    f"Invalid k value {k} for double norm term frequency. Values should be between 0 and 1.")
            self.f_tf = partial(self.double_norm_tf, k=k)
        else:
            raise ValueError(
                f"Unknown term frequency method {self.tf_method}. Choose on of {','.join(TF_METHOD_VALUES)}")

    @property
    def avgdl(self) -> float:
        """Average number of tokens per sentence in the extended news corpus.

        Returns
        -------
        avg_sentence_length: int
        """
        return self.config['default'].getfloat('avg_sentence_length')

    @property
    def tokenizer(self):
        """Configured tokenizer to tokenize the sentence.

        Returns
        -------
        tokenizer: sadedegel.bblock.WordTokenizer
        """
        return self.document.tokenizer

    @property
    def vocabulary(self):
        """Vocabulary dump built on extended news corpus.

        Returns
        -------
        vocabulary: sadedegel.bblock.vocabulary.Vocabulary
        """
        return self.tokenizer.vocabulary

    @classmethod
    def set_tf_function(cls, tf_type):
        raise DeprecationWarning("Function is depreciated.")

    @property
    def bert(self):
        return self._bert

    @bert.setter
    def bert(self, bert):
        self._bert = bert

    @property
    def input_ids(self):
        return self.tokenizer.convert_tokens_to_ids(self.tokens_with_special_symbols)

    @cached_property
    def tokens(self) -> List[Token]:
        return [t for t in self.tokenizer(self.text)]

    @property
    def tokens_with_special_symbols(self):
        return [Token('[CLS]')] + self.tokens + [Token('[SEP]')]

    def rouge1(self, metric) -> float:
        """ROUGE1 Relevance score of sentence wrt document

        Parameters
        ----------
        metric: str
            ROUGE1 metric type.

        Returns
        -------
            rouge1: float
        """
        return rouge1_score(
            flatten([[t.lower_ for t in sent] for sent in self.document if sent.id != self.id]),
            [t.lower_ for t in self], metric)

    @property
    def bm25(self) -> np.float32:
        """BM25 Relevance score of sentence wrt to document calculated based on configured arguments, tf and idf methods

        Returns
        -------
        bm25: float
        """

        tf = self.config['tf']['method']
        idf = self.config['idf']['method']
        drop_stopwords = self.config['default'].getboolean('drop_stopwords')
        lowercase = self.config['default'].getboolean('lowercase')
        drop_suffix = self.config['bert'].getboolean('drop_suffix')
        drop_punct = self.config['default'].getboolean('drop_punct')

        k1 = self.config['bm25'].getfloat('k1')
        b = self.config['bm25'].getfloat('b')

        delta = self.config['bm25'].getfloat('delta')

        return np.sum(self.get_bm25(tf, idf, k1, b, delta, drop_stopwords, lowercase, drop_suffix, drop_punct),
                      dtype=np.float32)

    @property
    def tfidf(self):
        """Sparse Tf-Idf vector representation calculated based on configured tf and idf methods

        Returns
        -------
        tfidf: numpy.ndarray (1, vocab_size)
        """
        tf = self.config['tf']['method']
        idf = self.config['idf']['method']
        drop_stopwords = self.config['default'].getboolean('drop_stopwords')
        lowercase = self.config['default'].getboolean('lowercase')
        drop_suffix = self.config['bert'].getboolean('drop_suffix')
        drop_punct = self.config['default'].getboolean('drop_punct')

        return self.get_tf(tf, drop_stopwords, lowercase, drop_suffix, drop_punct) * self.get_idf(idf, drop_stopwords,
                                                                                                  lowercase,
                                                                                                  drop_suffix,
                                                                                                  drop_punct)

    def get_tfidf(self, tf_method, idf_method, drop_stopwords=False, lowercase=False, drop_suffix=False,
                  drop_punct=False, **kwargs) -> np.ndarray:
        """Sparse Tf-Idf vector representation calculated based on user provided tf and idf methods

        Parameters
        ----------
        tf_method: str
            Term Frequency calculation method.
        idf_method: str
            Inverse Document Frequency calculation method.
        drop_stopwords: bool
            Drop stopwords from the sequence.
        lowercase: bool
            Lowerize all tokens in sequence.
        drop_suffix: bool
            Drop suffixes from sequence that is tokenized by sadedegel.bblock.BertTokenizer.
        drop_punct: bool
            Drop punctuation from the sequence.
        **kwargs: dict, optional

        Returns
        -------
        tfidf: numpy.ndarray (1, vocab_size)
        """
        return self.get_tf(tf_method, drop_stopwords, lowercase, drop_suffix, drop_punct, **kwargs) * self.get_idf(
            idf_method, drop_stopwords, lowercase, drop_suffix, drop_punct, **kwargs)

    @property
    def tf(self):
        """Sparse Term Frequency Vector

        Returns
        -------
        tf: numpy.ndarray (1, vocab_size)
        """
        return self.f_tf()

    @property
    def idf(self):
        """Sparse Inverse Document Frequency Vector calculated over vocabulary dump.

        Returns
        -------
        idf: numpy.ndarray (1, vocab_size)
        """
        v = np.zeros(len(self.vocabulary))

        for t in self.tokens:
            if not t.is_oov:
                v[t.id] = t.idf

        return v

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text

    def __len__(self):
        return len(self.tokens)

    def __eq__(self, s: str):
        return self.text == s  # no need for type checking, will return false for non-strings

    def __getitem__(self, token_ix):
        return self.tokens[token_ix]

    def __iter__(self):
        yield from self.tokens


class Document(TFImpl, IDFImpl, BM25Impl):
    """Document class is a sequence of Sentences objects. Access Sentences and Tokens that constitute the Document.
    Generate BoW or PreTrainedTransformer model based embeddings.

    Parameters
    ----------
    raw: str
        Raw string to be initialized as a Document object.
    builder: sadedegel.bblock.DocBuilder
        Builder class for lazy loading of models, initalization of configs and tokenizers. See `builder` attribute.

    Attributes
    ----------
    raw: str
        This is where raw document string is stored.
    builder: sadedegel.bblock.DocBuilder
        Sentence Boundary detection model (sbd), tokenizer and config are attributes of DocBuilder class. sadedegel.Doc object is an instance of sadedegel.bblock.DocBuilder.
        Importing sadedegel.Doc will instantiate such attributes once. Call of Doc will return an instance of sadedegel.bblock.Document that is initialized by referring above attributes.
    """
    def __init__(self, raw, builder):
        TFImpl.__init__(self)
        IDFImpl.__init__(self)
        BM25Impl.__init__(self)

        self.raw = raw
        self._spans = []
        self._sentences = []
        self.builder = builder
        self.config = self.builder.config

    @property
    def avgdl(self) -> float:
        """Average number of tokens per document"""
        return self.config['default'].getfloat('avg_document_length')

    @cached_property
    def _sents(self):
        if not self._sentences:
            _spans = [match.span() for match in re.finditer(r"\S+", self.raw)]
            self._spans = [Span(i, span, self) for i, span in enumerate(_spans)]

            if len(self._spans) > 0:
                y_pred = self.builder.sbd.predict((span.span_features() for span in self._spans))
            else:
                y_pred = []

            eos_list = [end for (start, end), y in zip(_spans, y_pred) if y == 1]
            if len(eos_list) > 0:
                for i, eos in enumerate(eos_list):
                    if i == 0:
                        self._sentences.append(Sentences(i, self.raw[:eos].strip(), self, self.builder.config))
                    else:
                        self._sentences.append(
                            Sentences(i, self.raw[eos_list[i - 1] + 1:eos].strip(), self, self.builder.config))

                if eos_list[-1] != len(self.raw):
                    self._sentences.append(
                        Sentences(len(self._sentences), self.raw[eos_list[-1] + 1:len(self.raw)], self,
                                  self.builder.config))
            else:
                self._sentences.append(Sentences(0, self.raw.strip(), self, self.builder.config))

        return self._sentences

    @cached_property
    def spans(self) -> List[Span]:
        """Span objects that constitute the Document

        Returns
        -------
        spans: List[sadedegel.bblock.Span]
        """
        _ = self._sents
        return self._spans

    @cached_property
    def tokens(self) -> List[Token]:
        """Token objects that constitute the Document

        Returns
        -------
        tokens: List[sadedegel.bblock.Token]
        """
        return [t for t in self.builder.tokenizer(self.raw)]

    @property
    def vocabulary(self):
        """Vocabulary that is used as reference for BoW based vector generation.

        Returns
        -------
        vocabulary: sadedegel.bblock.vocabulary.Vocabulary
        """
        return self.tokenizer.vocabulary

    @property
    def tokenizer(self):
        """Word tokenizer used to obtain Token objects that constitute the Document object.

        Returns
        -------
        tokenizer: sadedegel.bblock.WordTokenizer
        """
        return self.builder.tokenizer

    def __getitem__(self, key):
        return self._sents[key]

    def __iter__(self):
        return iter(self._sents)

    def __str__(self):
        return self.raw

    def __repr__(self):
        return self.raw

    def __len__(self):
        return len(self._sents)

    def max_length(self):
        """Maximum length of a sentence including special symbols."""
        return max(len(s.tokens_with_special_symbols) for s in self._sents)

    def padded_matrix(self, return_numpy=False, return_mask=True):
        """A zero-padded numpy.array or torch.tensor formed by sadedegel.tokenizer.BertTokenizer. Can be used for lower-level development of BERT pipelines with torch.
        One row for each sentence.
        One column for each token (pad 0 if length of sentence is shorter than the max length)

        Parameters
        -------
        return_numpy: bool
            Whether to return numpy.array or torch.tensor
        return_mask: bool
            Whether to return padding mask

        Returns
        -------
        padded_matrix: numpy.ndarray or torch.tensor
            Zero-padded tensor with type_ids of tokens encoded by sadedegel.tokenizer.BertTokenizer. (n_sequences, MAX_LEN)

        Raises
        ------
        ImportError
            If `transformers` module is not present in the environment.
        """
        max_len = self.max_length()

        if not return_numpy:
            try:
                import torch
            except ImportError:
                console.print(
                    ("Error in importing transformers module. "
                     "Ensure that you run 'pip install sadedegel[bert]' to use BERT features."))
                sys.exit(1)

            mat = torch.tensor([pad(s.input_ids, max_len) for s in self])

            if return_mask:
                return mat, (mat > 0).to(int)
            else:
                return mat
        else:
            mat = np.array([pad(s.input_ids, max_len) for s in self])

            if return_mask:
                return mat, (mat > 0).astype(int)
            else:
                return mat

    def get_pretrained_embedding(self, architecture: str, do_sents: bool):
        """Get dense document (or sentence) embeddings from a Transformer based pre-trained model hosted on HuggingFace Hub.
        The model will be downloaded to a local .cache directory if not locally available and used upon every call.
        GPU availiability is adviced for better speed.

        Parameters
        ----------
        architecture: str
            Transformer architecture to obtain embeddings from. Supported architectures are "bert_32k_cased", "bert_128k_cased", "bert_32k_uncased", "bert_128k_uncased", "distilbert"
        do_sents: bool
            If True, a matrix of sentence embeddings are returned. Defaults to False.

        Returns
        -------
        embeddings: numpy.ndarray
            Document (or sentence) embeddings. (1, emb_dim) for document. (n_sents, emb_dim) if do_sents=True.

        Raises
        ------
        ImportError
            If `sentence_transformers` module is not present in the environment.
        """
        try:
            from sentence_transformers import SentenceTransformer
            import transformers
            transformers.logging.set_verbosity_error()
        except ImportError as ie:
            console.print(
                ("Error in importing sentence_transformers module. "
                 "Ensure that you run 'pip install sadedegel[bert]' to use BERT and other transformer model features."))
            sys.exit(1)

        model_name = __transformer_model_mapper__.get(architecture)
        if model_name is None:
            raise ArchitectureNotFound(f"'{architecture}' is not a supported architecture type. "
                                       f"Try any among list of implemented architectures {list(__transformer_model_mapper__.keys())}")

        if DocBuilder.transformer_model is None:
            console.print(f"Loading \"{model_name}\"...")
            DocBuilder.transformer_model = TransformerModel(model_name, SentenceTransformer(model_name))
        elif DocBuilder.transformer_model.name != model_name:
            console.print(f"Changing configured transformer model of "
                          f"Doc from {DocBuilder.transformer_model.name} to {model_name}")
            DocBuilder.transformer_model = None
            self.get_pretrained_embedding(architecture=architecture, do_sents=do_sents)

        if do_sents:
            embeddings = DocBuilder.transformer_model.model.encode([s.text for s in self], show_progress_bar=False,
                                                                   batch_size=4)
        else:
            embeddings = DocBuilder.transformer_model.model.encode([self.raw], show_progress_bar=False)

        return embeddings

    @cached_property
    def bert_embeddings(self):
        """Get dense sentence embeddings from a BERT base cased model hosted on HuggingFace Hub.
        The model will be downloaded to a local .cache directory if not locally available and used upon every call.
        GPU availiability is adviced for better speed.

        Returns
        -------
        embeddings: numpy.ndarray
            Sentence embeddings. (n_sents, emb_dim)

        Raises
        ------
        ImportError
            If `sentence_transformers` module is not present in the environment.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as ie:
            console.print(
                ("Error in importing sentence_transformers module. "
                 "Ensure that you run 'pip install sadedegel[bert]' to use BERT and other transformer model features."))
            return ie

        if DocBuilder.bert_model is None:
            console.print("Loading \"dbmdz/bert-base-turkish-cased\"...")
            DocBuilder.bert_model = SentenceTransformer("dbmdz/bert-base-turkish-cased")

        embeddings = DocBuilder.bert_model.encode([s.text for s in self], show_progress_bar=False, batch_size=4)

        return embeddings

    @cached_property
    def bert_document_embedding(self):
        """Get dense document embedding from a BERT base cased model hosted on HuggingFace Hub.
        The model will be downloaded to a local .cache directory if not locally available and used upon every call.
        GPU availiability is adviced for better speed.

        Returns
        -------
        embeddings: numpy.ndarray
            Sentence embeddings. (n_sents, emb_dim)

        Raises
        ------
        ImportError
            If `sentence_transformers` module is not present in the environment.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as ie:
            console.print(
                ("Error in importing sentence_transformers module. "
                 "Ensure that you run 'pip install sadedegel[bert]' to use BERT and other transformer model features."))
            return ie

        if DocBuilder.bert_model is None:
            console.print("Loading \"dbmdz/bert-base-turkish-cased\"...")
            DocBuilder.bert_model = SentenceTransformer("dbmdz/bert-base-turkish-cased")

        embedding = DocBuilder.bert_model.encode([self.raw], show_progress_bar=False, batch_size=4)

        return embedding

    def get_tfidf(self, tf_method: str, idf_method: str, **kwargs):
        """Calculates and returns the tf-idf vector for the Document based on provided tf and idf methods.

        Parameters
        ----------
        tf_method: str
            Term Frequency calculation method.
        idf_method: str
            Inverse Document Frequency Calculation method.
        kwargs:

        Returns
        -------
            tfidf: numpy.ndarray
        """
        return self.get_tf(tf_method, **kwargs) * self.get_idf(idf_method, **kwargs)

    @property
    def tfidf(self):
        """Calculates and returns the tf-idf vector for the Document based on configured tf and idf methods.

        Returns
        -------
            tfidf: numpy.ndarray (1, vocab_size)
        """
        return self.tf * self.idf

    @property
    def tfidf_matrix(self):
        """Calculates and returns the tf-idf vector for the Sentences of the Document based on configured tf and idf methods.

        Returns
        -------
        tfidf_matrix: scipy.sparse.csr_matrix (n_sents, vocab_size)
        """

        indptr = [0]
        indices = []
        data = []

        for i in range(len(self)):
            sent_embedding = self[i].tfidf

            for idx in sent_embedding.nonzero()[0]:
                indices.append(idx)
                data.append(sent_embedding[idx])

            indptr.append(len(indices))

        lowercase = self.config['default'].getboolean('lowercase')

        if lowercase:
            dim = self.vocabulary.size
        else:
            dim = self.vocabulary.size_cs

        m = csr_matrix((data, indices, indptr), dtype=np.float32, shape=(len(self), dim))

        return m

    def from_sentences(self, sentences: List[str]):
        return self.builder.from_sentences(sentences)


class DocBuilder:
    transformer_model = None
    bert_model = None

    def __init__(self, **kwargs):

        self.config = load_config(kwargs)

        self.sbd = load_model()

        tokenizer_str = normalize_tokenizer_name(self.config['default']['tokenizer'])

        self.tokenizer = WordTokenizer.factory(tokenizer_str, emoji=self.config['tokenizer'].getboolean('emoji'),
                                               hashtag=self.config['tokenizer'].getboolean('hashtag'),
                                               mention=self.config['tokenizer'].getboolean('mention'),
                                               emoticon=self.config['tokenizer'].getboolean('emoticon'))

        Token.set_vocabulary(self.tokenizer.vocabulary)

        self.config['default']['avg_sentence_length'] = self.config[tokenizer_str]['avg_sentence_length']
        self.config['default']['avg_document_length'] = self.config[tokenizer_str]['avg_document_length']

        idf_method = self.config['idf']['method']

        if idf_method in IDF_METHOD_VALUES:
            Token.config = self.config
        else:
            raise ValueError(f"Unknown term frequency method {idf_method}. Choose on of {IDF_METHOD_VALUES}")

    def __call__(self, raw):

        if raw is not None:
            raw_stripped = raw.strip()
            d = Document(raw_stripped, self)


        else:
            raise Exception(f"{raw} document text can't be None")

        return d

    def from_sentences(self, sentences: List[str]):
        raw = "\n".join(sentences)

        d = Document(raw, self)
        for i, s in enumerate(sentences):
            d._sentences.append(Sentences(i, s, d, self.config))

        return d
