import warnings
from collections import defaultdict
from os.path import dirname
from pathlib import Path

import h5py
import numpy as np
from cached_property import cached_property
from rich.console import Console

from .util import tr_lower, normalize_tokenizer_name

console = Console()


class InvalidTokenizer(Exception):
    """Invalid tokenizer name"""


def vocabulary_file(tokenizer: str, verify_exists=True):
    normalized_name = normalize_tokenizer_name(tokenizer)

    if normalized_name not in ['bert', 'icu', 'simple']:
        raise InvalidTokenizer(
            (f"Currently only valid tokenizers are BERT, ICU Tokenizer for vocabulary generation."
             " {normalized_name} found"))

    vocab_file = Path(dirname(__file__)) / 'data' / normalized_name / 'vocabulary.hdf5'

    if not vocab_file.exists() and verify_exists:
        raise FileNotFoundError(f"Vocabulary file for {tokenizer} ({normalized_name}) tokenizer not found.")

    return vocab_file


class VocabularyCounter:
    def __init__(self, tokenizer, case_sensitive=True, min_tf=1, min_df=1):
        self.tokenizer = tokenizer

        self.doc_counter = defaultdict(set)
        self.doc_set = set()

        self.term_freq = defaultdict(int)

        self.min_tf = min_tf
        self.min_df = min_df
        self.case_sensitive = case_sensitive

    def inc(self, word: str, document_id: int, count: int = 1):
        if self.case_sensitive:
            w = word
        else:
            w = tr_lower(word)

        self.doc_counter[w].add(document_id)
        self.doc_set.add(document_id)
        self.term_freq[w] += count

    def add_word_to_doc(self, word: str, document_id: int):
        """Implemented for backward compatibility"""

        self.inc(word, document_id, 1)

    @property
    def vocabulary_size(self):
        return len(self.term_freq)

    @property
    def document_count(self):
        return len(self.doc_set)

    def prune(self):

        to_remove = []

        for w in self.term_freq:
            if self.term_freq[w] < self.min_tf or len(self.doc_counter[w]) < self.min_df:
                to_remove.append(w)

        for w in to_remove:
            del self.doc_counter[w]
            del self.term_freq[w]

        console.log(
            f"{len(to_remove)} terms (case sensitive={self.case_sensitive}) are pruned by tf (>= {self.min_tf}) or df filter(>= {self.min_df})")

        return self

    def df(self, w: str):
        if self.case_sensitive:
            return len(self.doc_counter[w])
        else:
            return len(self.doc_counter[tr_lower(w)])

    def tf(self, w: str):
        if self.case_sensitive:
            return self.term_freq[w]
        else:
            return self.term_freq[tr_lower(w)]

    def to_hdf5(self, w2v=None):
        with h5py.File(vocabulary_file(self.tokenizer, verify_exists=False), "a") as fp:
            if self.case_sensitive:
                group = fp.create_group("form_")
            else:
                group = fp.create_group("lower_")

            words = sorted(list(self.term_freq.keys()), key=lambda w: tr_lower(w))

            group.attrs['size'] = len(words)
            group.attrs['document_count'] = len(self.doc_set)
            group.attrs['tokenizer'] = self.tokenizer
            group.attrs['min_tf'] = self.min_tf
            group.attrs['min_df'] = self.min_df

            if w2v is not None:
                group.attrs['vector_size'] = w2v.vector_size

                group.create_dataset("vector", data=np.array(
                    [w2v[w] if w in w2v else np.zeros(w2v.vector_size) for w in words]).astype(
                    np.float32),
                                     compression="gzip",
                                     compression_opts=9)
                group.create_dataset("has_vector", data=np.array([w in w2v in w2v for w in words]),
                                     compression="gzip",
                                     compression_opts=9)

            group.create_dataset("word", data=words, compression="gzip", compression_opts=9)
            group.create_dataset("df", data=np.array([self.df(w) for w in words]), compression="gzip",
                                 compression_opts=9)
            group.create_dataset("tf", data=np.array([self.tf(w) for w in words]), compression="gzip",
                                 compression_opts=9)

        console.print(f"|D|: {self.document_count}, |V|: {self.vocabulary_size} (case sensitive={self.case_sensitive})")


class Vocabulary:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        self.file_name = vocabulary_file(tokenizer)
        self._df = None
        self._df_cs = None
        self._has_vector = None
        self._vector = None

        self.dword_cs = None
        self.dword = None

    @cached_property
    def size_cs(self) -> int:
        with h5py.File(self.file_name, "r") as fp:
            return fp['form_'].attrs['size']

    @cached_property
    def size(self) -> int:
        with h5py.File(self.file_name, "r") as fp:
            return fp['lower_'].attrs['size']

    def __len__(self):
        return self.size

    def id_cs(self, word: str, default: int = -1):
        if self.dword_cs is None:
            with h5py.File(self.file_name, "r") as fp:
                self.dword = dict((b.decode("utf-8"), i) for i, b in enumerate(list(fp['lower_']['word'])))
                self.dword_cs = dict((b.decode("utf-8"), i) for i, b in enumerate(list(fp['form_']['word'])))

        return self.dword_cs.get(word, default)

    def id(self, word: str, default: int = -1):
        if self.dword is None:
            with h5py.File(self.file_name, "r") as fp:
                self.dword = dict((b.decode("utf-8"), i) for i, b in enumerate(list(fp['lower_']['word'])))
                self.dword_cs = dict((b.decode("utf-8"), i) for i, b in enumerate(list(fp['form_']['word'])))

        return self.dword.get(tr_lower(word), default)

    def df(self, word: str):

        i = self.id(word)

        if i == -1:
            return 0
        else:
            if self._df is None:
                with h5py.File(self.file_name, "r") as fp:
                    self._df = np.array(fp['lower_']['df'])

            return self._df[i]

    def df_cs(self, word: str):

        i = self.id_cs(word)

        if i == -1:
            return 0
        else:
            if self._df_cs is None:
                with h5py.File(self.file_name, "r") as fp:
                    self._df_cs = np.array(fp['form_']['df'])

            return self._df_cs[i]

    def has_vector(self, word: str):
        with h5py.File(self.file_name, "r") as fp:
            if "has_vector" in fp['lower_']:
                i = self.id(word)

                if i == -1:
                    return False
                else:
                    if self._has_vector is None:
                        self._has_vector = np.array(fp['lower_']['has_vector'])

                    return self._has_vector[i]
            else:
                return False

    def vector(self, word: str):
        # TODO: Performance improvement required
        with h5py.File(self.file_name, "r") as fp:
            if "vector" in fp['lower_']:
                i = self.id(word)

                if i == -1:
                    return False
                else:
                    if self._vector is None:
                        self._vector = np.array(fp['lower_']['vector'])

                    return self._vector[i, :]
            else:
                return False

    @cached_property
    def document_count(self):
        with h5py.File(self.file_name, "r") as fp:
            return fp['form_'].attrs['document_count']
