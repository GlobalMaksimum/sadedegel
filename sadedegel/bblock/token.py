import unicodedata
from math import log

import numpy as np
from cached_property import cached_property

from .util import tr_lower, load_stopwords, deprecate, ConfigNotSet, VocabularyIsNotSet, WordVectorNotFound
from .vocabulary import Vocabulary

IDF_SMOOTH, IDF_PROBABILISTIC, IDF_UNARY = "smooth", "probabilistic", "unary"
IDF_METHOD_VALUES = [IDF_SMOOTH, IDF_PROBABILISTIC, IDF_UNARY]


class IDFImpl:
    def __init__(self):
        pass

    def get_idf(self, method=IDF_SMOOTH, drop_stopwords=False, lowercase=False, drop_suffix=False, drop_punct=False,
                **kwargs):

        if method not in IDF_METHOD_VALUES:
            raise ValueError(f"Unknown idf method ({method}). Choose one of {IDF_METHOD_VALUES}")

        if lowercase:
            v = np.zeros(self.vocabulary.size)
        else:
            v = np.zeros(self.vocabulary.size_cs)

        for t in self.tokens:
            if t.is_oov or (drop_stopwords and t.is_stopword) or (drop_suffix and t.is_suffix) or (
                    drop_punct and t.is_punct):
                continue

            if lowercase:
                if method == IDF_SMOOTH:
                    v[t.id] = t.smooth_idf
                elif method == IDF_UNARY:
                    v[t.id] = t.unary_idf
                else:
                    v[t.id] = t.prob_idf
            else:
                if method == IDF_SMOOTH:
                    v[t.id_cs] = t.smooth_idf_cs
                elif method == IDF_UNARY:
                    v[t.id] = t.unary_idf_cs
                else:
                    v[t.id_cs] = t.prob_idf_cs

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
    STOPWORDS = set(load_stopwords())
    vocabulary = None
    cache = {}

    @classmethod
    def _create_token(cls, word: str):
        token = object.__new__(cls)

        token.word = word
        token.lower_ = tr_lower(token.word)
        token.is_punct = all(unicodedata.category(c).startswith("P") for c in token.word)
        token.is_digit = token.word.isdigit()
        token.is_suffix = token.word.startswith('##')
        token.is_emoji = False
        token.is_hashtag = False
        token.is_mention = False

        return token

    def __new__(cls, word: str):

        if word not in cls.cache:
            cls.cache[word] = cls._create_token(word)

        return cls.cache[word]

    def __len__(self):
        return len(self.word)

    def __eq__(self, other):
        if type(other) == str:
            return self.word == other
        elif type(other) == Token:
            return self.word == other.word
        else:
            raise TypeError(f"Unknown comparison type with Token {type(other)}")

    @classmethod
    def set_vocabulary(cls, vocab: Vocabulary):
        Token.vocabulary = vocab
        Token.cache.clear()

    @classmethod
    def set_config(cls, config):
        Token.config = config

    @property
    def entry(self):
        deprecate("entry property is deprecated", (0, 20))

        return self

    @property
    def idf(self):
        if Token.config is None:
            raise ConfigNotSet("First run set_config.")
        else:
            if Token.config['idf']['method'] == IDF_SMOOTH:
                return self.smooth_idf
            elif Token.config['idf']['method'] == IDF_UNARY:
                return self.unary_idf
            else:
                return self.prob_idf

    @property
    def smooth_idf(self):
        if Token.vocabulary is None:
            raise VocabularyIsNotSet("First run set_vocabulary")
        else:
            return log(self.vocabulary.document_count / (1 + self.df)) + 1

    @property
    def smooth_idf_cs(self):
        if Token.vocabulary is None:
            raise VocabularyIsNotSet("First run set_vocabulary")
        else:
            return log(self.vocabulary.document_count / (1 + self.df_cs)) + 1

    @property
    def unary_idf(self):
        if Token.vocabulary is None:
            raise VocabularyIsNotSet("First run set_vocabulary")
        else:
            return int(self.df > 0)

    @property
    def unary_idf_cs(self):
        if Token.vocabulary is None:
            raise VocabularyIsNotSet("First run set_vocabulary")
        else:
            return int(self.df_cs > 0)

    @property
    def prob_idf(self) -> float:
        if Token.vocabulary is None:
            raise VocabularyIsNotSet("First run set_vocabulary")
        else:
            df = self.df + 1
            return log((self.vocabulary.document_count - df) / df)

    @property
    def prob_idf_cs(self) -> float:
        if Token.vocabulary is None:
            raise VocabularyIsNotSet("First run set_vocabulary")
        else:
            df = self.df_cs + 1
            return log((self.vocabulary.document_count - df) / df)

    @property
    def id_cs(self) -> int:
        """Vocabulary identifier"""

        if Token.vocabulary is None:
            raise VocabularyIsNotSet("First run set_vocabulary")
        else:
            return self.vocabulary.id_cs(self.word, -1)

    @property
    def id(self) -> int:
        """Vocabulary identifier (incase sensitive, aka lowercase all tokens)"""

        if Token.vocabulary is None:
            raise VocabularyIsNotSet("First run set_vocabulary")
        else:
            return self.vocabulary.id(self.word, -1)

    @property
    def is_oov(self) -> bool:
        """Return False if token type is not a part of vocabulary"""

        return self.id == -1

    @property
    def is_stopword(self) -> bool:
        return self.lower_ in Token.STOPWORDS

    @property
    def df(self) -> int:
        return self.vocabulary.df(self.word)

    @property
    def df_cs(self) -> int:
        """case sensitive document frequency"""
        return self.vocabulary.df_cs(self.word)

    @property
    def has_vector(self) -> bool:
        return self.vocabulary.has_vector(self.word)

    @property
    def vector(self) -> np.ndarray:
        if self.has_vector:
            return self.vocabulary.vector(self.word)
        else:
            raise WordVectorNotFound(self.word)

    @cached_property
    def shape(self) -> str:
        return word_shape(self.word)

    def __str__(self):
        return self.word

    def __repr__(self):
        return self.word
