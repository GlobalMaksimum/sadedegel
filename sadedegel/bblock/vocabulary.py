import unicodedata
from os.path import dirname
from pathlib import Path
from math import log
import warnings
from json import dump, load
from sadedegel.bblock.util import tr_lower


class Vocabulary:
    tokens = {}
    size = None
    tokenizer = "bert"

    @classmethod
    def token(cls, word):
        return Vocabulary.tokens.get(tr_lower(word), None)

    @classmethod
    def save(cls):
        words = list(Vocabulary.tokens.values())

        with open(Path(dirname(__file__)) / 'data' / 'vocabulary.json', "w") as fp:
            dump(dict(size=Vocabulary.size, tokenizer=Vocabulary.tokenizer, words=words), fp, ensure_ascii=False)

    @classmethod
    def load(cls):
        with open(Path(dirname(__file__)) / 'data' / 'vocabulary.json') as fp:
            json = load(fp)

        vocab = Vocabulary()
        Vocabulary.size = json['size']
        Vocabulary.tokenizer = json['tokenizer']

        for w in json['words']:
            Vocabulary.tokens[w['word']] = w

        return vocab


def get_vocabulary(tokenizer):
    try:
        if tokenizer.__name__ == "BertTokenizer":
            return Vocabulary.load()
        else:
            warnings.warn("Vocabulary is only available for BertTokenizer.")
            return None
    except FileNotFoundError:
        warnings.warn("vocabulary.bin is not available. Some functionalities my fail")
        return None


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
    vocabulary = None
    cache = {}

    @classmethod
    def _get_cache(cls, word):
        token = Token.cache.get(word, None)

        return token

    @classmethod
    def set_vocabulary(cls, tokenizer):
        Token.cache.clear()
        Token.vocabulary = get_vocabulary(tokenizer)

        return Token.vocabulary

    def __new__(cls, word, *args, **kwargs):
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

        if Token.vocabulary:
            token = Token.vocabulary.token(self.lower_)

            if not token:
                self.is_oov = True
            else:
                self.is_oov = False
                self.id = token['id']
                self.df = token['df']
                self.n_document = token['n_document']

            self.f_idf = self.smooth_idf
        else:
            self.f_idf = self.none_idf

    def smooth_idf(self):
        return log(self.n_document / (1 + self.df)) + 1

    def none_idf(self):
        warnings.warn("Vocabulary is only available for BertTokenizer.")
        return None

    @property
    def idf(self):
        return self.f_idf()
