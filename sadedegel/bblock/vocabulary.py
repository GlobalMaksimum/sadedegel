from os.path import dirname
from pathlib import Path
from dataclasses import dataclass
from math import log
from json import dump, load
from sadedegel.bblock.util import tr_lower
from sadedegel.bblock.word_tokenizer_helper import puncts

@dataclass
class Token:
    id: int
    word: str
    df: int
    n_document: int

    @property
    def is_punct(self):
        return self.word in puncts

    @property
    def is_digit(self):
        return self.word.isdigit()

    @property
    def shape(self):
        if self.is_digit:
            shape = 'd' * len(self.word)
        else:
            shape = ''
            for char in self.word:
                if char.isupper():
                    shape += 'X'
                else:
                    shape += 'x'

        return shape

    @property
    def idf(self):
        return log(self.n_document / (1 + self.df)) + 1

    @classmethod
    def from_dict(cls, d: dict):
        return Token(d['id'], d['word'], d['df'], d['n_document'])


class Vocabulary:
    tokens = {}
    size = None

    @classmethod
    def token(cls, word):
        return Vocabulary.tokens.get(tr_lower(word), None)

    @classmethod
    def save(cls):
        words = []

        for t in Vocabulary.tokens.values():
            words.append(dict(id=t.id, word=t.word, df=t.df, n_document=t.n_document))

        with open(Path(dirname(__file__)) / 'data' / 'vocabulary.json', "w") as fp:
            dump(dict(size=Vocabulary.size, tokenizer="bert", words=words), fp, ensure_ascii=False)

    @classmethod
    def load(cls):
        with open(Path(dirname(__file__)) / 'data' / 'vocabulary.json') as fp:
            json = load(fp)

        vocab = Vocabulary()
        Vocabulary.size = json['size']

        for w in json['words']:
            Vocabulary.tokens[w['word']] = Token.from_dict(w)

        return vocab


def get_vocabulary(tokenizer):
    try:
        return Vocabulary.load()
    except FileNotFoundError:
        import warnings
        warnings.warn("vocabulary.bin is not available. Some functionalities my fail")
        return None
