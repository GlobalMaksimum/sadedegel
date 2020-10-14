import unicodedata

from math import log
from .util import tr_lower
from .vocabulary import Vocabulary
from .word_tokenizer import BertTokenizer


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
    _vocabulary = None
    cache = {}
    idf_type = "smooth"

    @classmethod
    def vocabulary(cls):
        if cls._vocabulary is None:
            cls.set_vocabulary(BertTokenizer)

        return cls._vocabulary

    @classmethod
    def reset(cls):
        cls.cache.clear()
        cls._vocabulary = None

    @classmethod
    def _get_cache(cls, word):
        token = Token.cache.get(word, None)

        return token

    @classmethod
    def set_vocabulary(cls, tokenizer):
        cls.cache.clear()
        cls._vocabulary = Vocabulary.load(tokenizer.__name__)

    def __new__(cls, word, *args, **kwargs):
        if cls._vocabulary is None:
            cls._vocabulary = Vocabulary.load(BertTokenizer.__name__)

        cached = cls._get_cache(word)

        if cached:
            return cached

        token = super(Token, cls).__new__(cls)

        return token

    def __init__(self, word):
        if self in self.cache:
            return

        self.cache[word] = self

        self.word = word
        self.lower_ = tr_lower(word)
        self.is_punct = all(unicodedata.category(c).startswith("P") for c in word)
        self.is_digit = word.isdigit()
        self.shape = word_shape(word)
        self.f_idf = self.get_idf_func

        self.entry = None

    def smooth_idf(self):
        return log(Token._vocabulary.document_count / (1 + self.df)) + 1

    def prob_idf(self):
        return log((Token._vocabulary.document_count - self.df) / self.df)

    @property
    def idf(self):
        return self.f_idf()

    @property
    def id(self):
        if self.entry is None:
            self.entry = Token._vocabulary[self.word]

        return self.entry.id

    @property
    def is_oov(self):
        return self.id == -1

    @property
    def df(self):
        if self.entry is None:
            self.entry = Token._vocabulary[self.word]

        return self.entry.df

    def get_idf_func(self):
        idf_config = Token.idf_type
        idf_funcs = {'smooth': self.smooth_idf(),
                     'probabilistic': self.prob_idf()}
        return idf_funcs[idf_config]

    @classmethod
    def set_idf_function(cls, idf_type):
        Token.idf_type = idf_type
