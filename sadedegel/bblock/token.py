import unicodedata

from math import log
import numpy as np
from .util import tr_lower, load_stopwords

IDF_SMOOTH, IDF_PROBABILISTIC = "smooth", "probabilistic"
IDF_METHOD_VALUES = [IDF_SMOOTH, IDF_PROBABILISTIC]


class IDFImpl:
    def __init__(self):
        pass

    def get_idf(self, method=IDF_SMOOTH, drop_stopwords=False, lowercase=False, drop_suffix=False, drop_punct=False,
                **kwargs):

        if method not in IDF_METHOD_VALUES:
            raise ValueError(f"Unknown idf method ({method}). Choose one of {IDF_METHOD_VALUES}")

        v = np.zeros(len(self.vocabulary))

        if lowercase:
            tokens = [tr_lower(t) for t in self.tokens]
        else:
            tokens = self.tokens

        for token in tokens:
            t = self.vocabulary[token]
            if t.is_oov or (drop_stopwords and t.is_stopword) or (drop_suffix and t.is_suffix) or (
                    drop_punct and t.is_punct):
                continue

            if method == IDF_SMOOTH:
                v[t.id] = t.smooth_idf
            else:
                v[t.id] = t.prob_idf

        return v

    @property
    def idf(self):
        idf = self.config['idf']['method']
        drop_stopwords = self.config['default'].getboolean('drop_stopwords')
        lowercase = self.config['default'].getboolean('lowercase')
        drop_suffix = self.config['bert'].getboolean('drop_suffix')
        drop_punct = self.config['default'].getboolean('drop_punct')

        return self.get_idf(idf, drop_stopwords, lowercase, drop_suffix, drop_punct)


def word_shape(text):
    if len(text) >= 100:
        return "LONG"
    shape = []
    last = ""
    shape_char = ""
    seq = 0
    for char in text:
        if char.isalpha():
            if char.isupper():
                shape_char = "X"
            else:
                shape_char = "x"
        elif char.isdigit():
            shape_char = "d"
        else:
            shape_char = char
        if shape_char == last:
            seq += 1
        else:
            seq = 0
            last = shape_char
        if seq < 4:
            shape.append(shape_char)
    return "".join(shape)


class Token:
    config = None
    STOPWORDS = None
    cache = {}

    def __init__(self, entry):
        if self in self.cache:
            return

        if isinstance(entry, str):
            self.cache[entry] = self

            self.word = entry
            self.lower_ = tr_lower(self.word)
            self.is_punct = all(unicodedata.category(c).startswith("P") for c in self.word)
            self.is_digit = self.word.isdigit()
            self.is_suffix = self.word.startswith('##')
            self.shape = word_shape(self.word)

            self._entry = None
        else:
            self.cache[entry.word] = self

            self.word = entry.word
            self.lower_ = tr_lower(self.word)
            self.is_punct = all(unicodedata.category(c).startswith("P") for c in self.word)
            self.is_digit = self.word.isdigit()
            self.is_suffix = self.word.startswith('##')
            self.shape = word_shape(self.word)

            self._entry = entry

    @property
    def entry(self):
        if self._entry is None:
            raise ValueError(f"Token is initialized with a str object. Initialized with Vocabulary entry")

        return self._entry

    @property
    def idf(self):
        if Token.config['idf']['method'] == IDF_SMOOTH:
            return self.smooth_idf
        else:
            return self.prob_idf

    @property
    def smooth_idf(self):
        return log(self.entry.vocabulary.document_count / (1 + self.df)) + 1

    @property
    def prob_idf(self):
        return log((self.entry.vocabulary.document_count - self.df) / self.df)

    @property
    def id(self):
        return self.entry.id

    @property
    def is_oov(self):
        return self.id == -1

    @property
    def is_stopword(self):
        if Token.STOPWORDS is None:
            Token.STOPWORDS = set(load_stopwords())

        return self.lower_ in Token.STOPWORDS

    @property
    def df(self):
        return self.entry.df

    @property
    def df_cs(self):
        """case sensitive document frequency"""
        return self.entry.df_cs

    def __str__(self):
        return self.word

    def __repr__(self):
        return self.word
