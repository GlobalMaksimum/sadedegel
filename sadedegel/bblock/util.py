import sys
import warnings
from collections import defaultdict
from os.path import dirname
from pathlib import Path
from typing import List

import numpy as np
from rich.console import Console

from ..about import __version__

console = Console()

__tr_upper__ = "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"
__tr_lower__ = "abcçdefgğhıijklmnoöprsştuüvyz"

__tr_lower_abbrv__ = ['hz.', 'dr.', 'prof.', 'doç.', 'org.', 'sn.', 'st.', 'mah.', 'mh.', 'sok.', 'sk.', 'alb.', 'gen.',
                      'av.', 'ist.', 'ank.', 'izm.', 'm.ö.', 'k.k.t.c.']


def tr_lower(s: str) -> str:
    if "I" in s or "İ" in s:
        return s.replace("I", "ı").replace("İ", "i").lower()
    else:
        return s.lower()


def tr_upper(s: str) -> str:
    return s.replace("i", "İ").upper()


def space_pad(token):
    return " " + token + " "


def pad(l, padded_length):
    return l + [0 for _ in range(padded_length - len(l))]


def flatten(l2: List[List]):
    flat = []
    for l in l2:
        for e in l:
            flat.append(e)

    return flat


def is_eos(span, sentences: List[str]) -> int:
    start = 0
    eos = []
    for s in sentences:
        idx = span.doc.raw.find(s, start) + len(s) - 1
        eos.append(idx)

        start = idx

    b, e = span.value

    for idx in eos:
        if b <= idx <= e:
            return 1

    return 0


def select_layer(bert_out: tuple, layers: List[int], return_cls: bool) -> np.ndarray:
    """Selects and averages layers from BERT output.

    Parameters:
        bert_out: tuple
            Tuple containing output of 12 intermediate layers after feeding a document.

        layers: List[int]
            List that contains which layer to choose. max = 11, min = 0.

        return_cls: bool
            Whether to use CLS token embedding as sentence embedding instead of averaging token embeddings.

    Returns:
        numpy.ndarray (n_sentences, embedding_size) Embedding size if default to 768.

    """
    n_layers = len(layers)
    n_sentences = bert_out[0].shape[0]
    n_tokens = bert_out[0].shape[1]

    if not (min(layers) > -1 and max(layers) < 12):
        raise Exception(f"Value for layer should be in 0-11 range")

    if return_cls:
        cls_matrix = np.zeros((n_layers, n_sentences, 768))
        l_ix = 0
        for l, layer in enumerate(bert_out):
            if l not in layers:
                continue
            else:
                l_ix = l_ix + 1
            for s, sentence in enumerate(layer):
                cls_tensor = sentence[0].numpy()
                cls_matrix[l_ix - 1, s, :] = cls_tensor
        layer_mean_cls = np.mean(cls_matrix, axis=0)

        return layer_mean_cls

    else:
        token_matrix = np.zeros((n_layers, n_sentences, n_tokens - 2, 768))
        for l, layer in enumerate(bert_out):
            l_ix = 0
            if l not in layers:
                continue
            else:
                l_ix = l_ix + 1
            for s, sentence in enumerate(layer):
                for t, token in enumerate(sentence[1:-1]):  # Exclude [CLS] and [SEP] embeddings
                    token_tensor = sentence[t].numpy()
                    token_matrix[l_ix - 1, s, t, :] = token_tensor

        tokenwise_mean = np.mean(token_matrix, axis=2)
        layer_mean_token = np.mean(tokenwise_mean, axis=0)

        return layer_mean_token


def normalize_tokenizer_name(tokenizer_name, raise_on_error=False):
    normalized = tokenizer_name.lower().replace(' ', '').replace('-', '').replace('tokenizer', '')

    if normalized not in ['bert', 'simple', 'icu']:
        msg = f"Invalid tokenizer {tokenizer_name} ({normalized}). Valid values are bert, simple"
        if raise_on_error:
            raise ValueError(msg)
        else:
            warnings.warn(msg, UserWarning, stacklevel=3)

    return normalized


def to_config_dict(kw: dict):
    d = defaultdict(lambda: dict())
    for k, v in kw.items():
        if '__' not in k:  # default section
            d['default'][k] = v
        else:
            section, key = k.split('__')

            d[section][key] = v

    return d


def load_stopwords(base_path=None):
    """ Return Turkish stopwords as list from file. """
    if base_path is None:
        base_path = dirname(__file__)

    text_path = Path(base_path) / "data" / "stop-words.txt"

    with open(text_path, "r") as fp:
        stopwords = fp.readlines()

    stopwords = [s.rstrip() for s in stopwords]

    return stopwords


def deprecate(message: str, eol_version: tuple, post_message: str = None):
    current = tuple([int(v) for v in __version__.split('.')])

    if current >= eol_version:
        console.print(f"[red]{message}[/red]. {post_message}")
        sys.exit(1)
    else:
        console.print(
            f"{message}, will be [magenta]dropped[/magenta] by {'.'.join(map(str, eol_version))}. {post_message}")


class ConfigNotSet(Exception):
    """Used when configuration is not set although it is expected"""


class VocabularyIsNotSet(Exception):
    """Raised when vocabulary class variable is not set"""


class WordVectorNotFound(Exception):
    """Word Vector is not available for word"""

    def __init__(self, w: str):
        self.word = w

        super().__init__(f"Word Vector not found for {self.word}")
