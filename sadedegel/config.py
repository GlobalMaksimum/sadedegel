__all__ = ['set_config', 'get_config', 'describe_config', 'get_all_configs']

from typing import Any
from functools import wraps
from collections import namedtuple
from contextlib import contextmanager
import warnings
from .bblock.doc import Sentences
from .bblock.token import Token

Configuration = namedtuple("Configuration", "config, description, valid_values")

configs = {
    "word_tokenizer": Configuration(config="word_tokenizer",
                                    description="Change the default word tokenizer used by sadedegel",
                                    valid_values=None),
    "idf": Configuration(config="idf",
                         description="Change default idf function used by sadedegel",
                         valid_values=['smooth', 'probabilistic'])
}


def check_config(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        config = args[0]
        if config not in configs:
            raise Exception((f"{config} is not a valid configuration for sadedegel."
                             "Use sadedegel.get_all_configs() to access list of valid configurations."))
        return f(*args, **kwds)

    return wrapper


def check_value(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        config, value = args[0], args[1]
        cfg = configs.get(config, None)

        if cfg:
            if cfg.config == 'idf':
                if value not in cfg.valid_values:
                    raise Exception(
                        f"{value} is not a valid value for {config}. Choose one of {', '.join(cfg.valid_values)}")
        else:
            raise Exception((f"{config} is not a valid configuration for sadedegel."
                             "Use sadedegel.get_all_configs() to access list of valid configurations."))

        return f(*args, **kwds)

    return wrapper


@check_value
def set_config(config: str, value: Any):
    if config == "word_tokenizer":
        Sentences.set_word_tokenizer(value)
    if config == 'idf':
        Token.set_idf_function(value)


@contextmanager
def tokenizer_context(tokenizer_name, warning=False):
    current = Sentences.tokenizer.__name__

    if warning and current != tokenizer_name:
        warnings.warn(f"Changing tokenizer to {tokenizer_name}")

    try:
        set_config("word_tokenizer", tokenizer_name)
        yield
    finally:
        set_config("word_tokenizer", current)


@contextmanager
def idf_context(idf_type, warning=False):
    current = Token.idf_type

    if warning and current != idf_type:
        warnings.warn(f"Changing idf function to {idf_type}")

    try:
        set_config('idf', idf_type)
        yield
    finally:
        set_config('idf', current)


@check_config
def get_config(config: str):  # pylint: disable=inconsistent-return-statements
    if config == "word_tokenizer":
        return Sentences.tokenizer.__name__
    if config == "idf":
        return Token.idf_type


@check_config
def describe_config(config: str, print_desc=False):  # pylint: disable=inconsistent-return-statements
    if configs[config].valid_values is not None:
        valid_values_fragment = "\n\nValid values are\n" + "\n".join(configs[config].valid_values)
    else:
        valid_values_fragment = ""

    config_doc = f"{configs[config].description}{valid_values_fragment}"

    if print_desc:
        print(config_doc)
    else:
        return config_doc


def get_all_configs():
    return configs
