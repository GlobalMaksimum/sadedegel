import unicodedata
import os
from os.path import dirname
from pathlib import Path
from math import log
from json import dump, load
from sadedegel.bblock.util import tr_lower
from sadedegel.bblock.word_tokenizer_helper import puncts, normalize_word_tokenizer_name

class Vocabulary:
    tokens = {} # dict of dicts in form of "<word_token>":{"<token_attr1":"token_attr1_val", ...}
    size = None
    default_tokenizer = "bert"
    tokenizer = None # set in load based on loaded vocabulary

    @classmethod
    def token(cls, word):
        return Vocabulary.tokens.get(tr_lower(word), None)

    @classmethod
    def save(cls, word_tokenizer_name: str):
        words = list(Vocabulary.tokens.values())

        path = Vocabulary._get_filepath(word_tokenizer_name)
        os.makedirs(path.parent, exist_ok=True) # create folder housing vocabulary.json

        with open(path, "w") as fp:
            dump(dict(size=Vocabulary.size, tokenizer=word_tokenizer_name, words=words), fp, ensure_ascii=False)

    @classmethod
    def load(cls, word_tokenizer_name: str):
        with open(Vocabulary._get_filepath(word_tokenizer_name)) as fp:
            json = load(fp)

        vocab = Vocabulary()
        Vocabulary.size = json['size']
        Vocabulary.tokenizer = json['tokenizer']

        for w in json['words']:
            Vocabulary.tokens[w['word']] = w

        return vocab

    @classmethod
    def _get_filepath(cls, word_tokenizer_name: str):
        tok_name = normalize_word_tokenizer_name(word_tokenizer_name)
        p = Path(dirname(__file__))
        return p / 'data' / tok_name / 'vocabulary.json'

def get_vocabulary(tokenizer):
    try:
        return Vocabulary.load(tokenizer.__name__)
    except FileNotFoundError:
        import warnings
        warnings.warn("{} is not available. \
                      Some functionalities may fail.".format(str(Vocabulary._get_filepath(tokenizer.__name__))))
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
    def set_vocabulary(cls, tokenizer=None):
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
        print("Set vocabulary first using `set_vocabulary` class method.")
        return None

    @property
    def idf(self):
        return self.f_idf()
