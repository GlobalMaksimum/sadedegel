from os.path import dirname
from pathlib import Path
import warnings
from dataclasses import dataclass, asdict
from json import dump, load
from collections import defaultdict

from .word_tokenizer import normalize_tokenizer_name
from .util import tr_lower


@dataclass
class Entry:
    id: int
    word: str
    df: int
    df_cs: int


class Vocabulary:
    vocabularies = {}

    @staticmethod
    def factory(tokenizer_name: str):
        normalized_name = normalize_tokenizer_name(tokenizer_name)

        if normalized_name not in Vocabulary.vocabularies:
            Vocabulary.vocabularies[normalized_name] = Vocabulary(normalized_name)

        return Vocabulary.vocabularies[normalized_name]

    def __init__(self, tokenizer_name: str):
        self.tokenizer_name = tokenizer_name
        self.entries = {}
        self.initialized = False
        self.document_count = -1

        self.doc_counter = defaultdict(set)
        self.doc_counter_case_sensitive = defaultdict(set)
        self.doc_set = set()

    @property
    def size(self):
        return len(self.entries)

    def __len__(self):
        return self.size

    def add_word_to_doc(self, word, doc_identifier):
        self.doc_counter[tr_lower(word)].add(doc_identifier)
        self.doc_counter_case_sensitive[word].add(doc_identifier)
        self.doc_set.add(doc_identifier)

    def build(self, min_df=1):
        i = 0
        for word in self.doc_counter_case_sensitive:
            if len(self.doc_counter[tr_lower(word)]) >= min_df:
                self.entries[word] = Entry(i, word, len(self.doc_counter[tr_lower(word)]),
                                           len(self.doc_counter_case_sensitive[word]))
                i += 1

        self.document_count = len(self.doc_set)
        self.initialized = True

    def save(self):
        if not self.initialized:
            raise Exception("Call build to initialize vocabulary")

        with open(Vocabulary._get_filepath(self.tokenizer_name), "w") as fp:
            dump(dict(size=len(self), document_count=self.document_count, tokenizer=self.tokenizer_name,
                      words=[asdict(e) for e in self.entries.values()]), fp,
                 ensure_ascii=False)

    @staticmethod
    def load(tokenizer_name: str):
        normalized_name = normalize_tokenizer_name(tokenizer_name)

        vocab = Vocabulary.factory(normalized_name)

        if not vocab.initialized:
            with open(Vocabulary._get_filepath(tokenizer_name), "r") as fp:
                json = load(fp)

            for d in json['words']:
                vocab.entries[d['word']] = Entry(d['id'], d['word'], d['df'], d['df_cs'])

            vocab.document_count = json['document_count']
            vocab.initialized = True

        return vocab

    def __getitem__(self, word):
        if not self.initialized:
            raise Exception(("Vocabulary instance is not initialize. "
                             "Use load for built in vocabularies, use build for manual vocabulary build."))
        else:
            entry = self.entries.get(word, None)

            if entry:
                return entry
            else:
                return Entry(-1, None, 0, 0)  # OOV has a document frequency of 0 by convention

    @staticmethod
    def _get_filepath(tokenizer_name: str):
        tok_name = normalize_tokenizer_name(tokenizer_name)
        p = Path(dirname(__file__))

        return p / 'data' / tok_name / 'vocabulary.json'
