import unicodedata

from math import log
from .util import tr_lower


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
            self.shape = word_shape(self.word)

            self._entry = None
        else:
            self.cache[entry.word] = self

            self.word = entry.word
            self.lower_ = tr_lower(self.word)
            self.is_punct = all(unicodedata.category(c).startswith("P") for c in self.word)
            self.is_digit = self.word.isdigit()
            self.shape = word_shape(self.word)

            self._entry = entry

    @property
    def entry(self):
        if self._entry is None:
            raise ValueError(f"Token is initialized with a str object. Initialized with Vocabulary entry")

        return self._entry

    def smooth_idf(self):
        return log(self.entry.vocabulary.document_count / (1 + self.df)) + 1

    @property
    def idf(self):
        return self.smooth_idf()

    @property
    def id(self):
        return self.entry.id

    @property
    def is_oov(self):
        return self.id == -1

    @property
    def df(self):
        return self.entry.df

    @property
    def df_cs(self):
        """case sensitive document frequency"""
        return self.entry.df_cs
