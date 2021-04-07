__all__ = ['tokenizer_context', 'config_context', 'idf_context', 'tf_context', 'load_config']

import warnings
from collections import defaultdict
from configparser import ConfigParser
from contextlib import contextmanager
from os.path import dirname
from pathlib import Path

from rich.console import Console
from rich.table import Table


@contextmanager
def tokenizer_context(tokenizer_name, warning=False):
    from .bblock import DocBuilder  # pylint: disable=import-outside-toplevel

    if warning:
        warnings.warn(f"Changing tokenizer to {tokenizer_name}")

    yield DocBuilder(tokenizer=tokenizer_name)


@contextmanager
def config_context(**kwargs):
    from .bblock import DocBuilder  # pylint: disable=import-outside-toplevel

    yield DocBuilder(**kwargs)


@contextmanager
def idf_context(idf_type, warning=False):  # pylint: disable=unused-argument
    from .bblock import DocBuilder  # pylint: disable=import-outside-toplevel

    yield DocBuilder(idf__method=idf_type)


@contextmanager
def tf_context(tf_type, warning=False):  # pylint: disable=unused-argument
    from .bblock import DocBuilder  # pylint: disable=import-outside-toplevel

    yield DocBuilder(tf__method=tf_type)


def to_config_dict(kw: dict):
    d = defaultdict(dict)
    for k, v in kw.items():
        if '__' not in k:  # default section
            d['default'][k] = v
        else:
            section, key = k.split('__')

            d[section][key] = v

    return d


def load_config(kwargs: dict = None):
    config = ConfigParser()
    config.read([Path(dirname(__file__)) / 'default.ini', Path("~/.sadedegel/user.ini").expanduser()])

    if kwargs:
        config_dict = to_config_dict(kwargs)
        config.read_dict(config_dict)

    return config


def show_config(config, section=None):
    descriptions = {"default__tokenizer": "Word tokenizer to use",
                    "default__drop_stopwords": ("Whether to drop stopwords in various calculations. "
                                                "Such as, tfidf, bm25, etc."),
                    "default__lowercase": "Whether to use lowercased form rather than form itself.",
                    "tokenizer__hashtag": "enable/disable hashtag (#sadedegel) handler in word tokenizer",
                    "tokenizer__mention": "enable/disable mention (@sadedegel) handler in word tokenizer",
                    "tokenizer__emoji": "enable/disable emoji (🍰) handler in word tokenizer",
                    "default__drop_punct": ("Whether to drop punctuations in various calculations. "
                                            "Such as, tfidf, bm25, etc."),
                    "tf__method": "Method used in term frequency calculation",
                    "tf__double_norm_k": "Smooth parameter used by double norm term frequency method.",
                    "idf__method": "Method used in Inverse Document Frequency calculation",
                    "bert__avg_document_length": "Average number of tokens in a bert tokenized document.",
                    "bert__avg_sentence_length": "Average number of tokens in a bert tokenized sentences.",
                    "icu__avg_document_length": "Average number of tokens in a icu tokenized document.",
                    "icu__avg_sentence_length": "Average number of tokens in a icu tokenized sentences.",
                    "bert__drop_suffix": ("Whether to drop BERT generated suffixes in various calculations. "
                                          "Such as, tfidf, bm25, etc."),
                    "simple__avg_document_length": "Average token count in a simple tokenizer tokenized document.",
                    "simple__avg_sentence_length": "Average token count in a simple tokenizer tokenized sentences.",
                    "bm25__k1": "BM25 k1 parameter as defined in https://en.wikipedia.org/wiki/Okapi_BM25",
                    "bm25__b": "BM25 b parameter as defined in https://en.wikipedia.org/wiki/Okapi_BM25",
                    "bm25__delta": "BM25+ delta parameter as defined in https://en.wikipedia.org/wiki/Okapi_BM25",
                    }

    default_config = ConfigParser()
    default_config.read([Path(dirname(__file__)) / 'default.ini'])

    console = Console()

    table = Table(show_header=True, header_style="bold #2070b2")

    table.add_column("section")
    table.add_column("parameter_name")
    table.add_column("current_value")
    table.add_column("default_value")
    table.add_column("description", width=50)

    for sec in config.sections():
        if sec == section or section is None:
            for k in config[sec]:
                if config[sec][k] != default_config[sec][k]:
                    table.add_row(sec, k, f"[orange1]{config[sec][k]}[/orange1]", default_config[sec][k],
                                  descriptions.get(f"{sec}__{k}", ""))
                else:
                    table.add_row(sec, k, f"{config[sec][k]}", default_config[sec][k],
                                  descriptions.get(f"{sec}__{k}", ""))

    console.print(table)
