__all__ = ['set_config', 'get_config', 'describe_config', 'get_all_configs']

from typing import Any
from functools import wraps
from collections import namedtuple
from contextlib import contextmanager
import warnings

Configuration = namedtuple("Configuration", "config, description, valid_values")

configs = {
    "word_tokenizer": Configuration(config="word_tokenizer",
                                    description="word_tokenizer is used to split sentences into words.",
                                    valid_values=['bert', 'simple']),

    "bert_device": Configuration(config="bert_device",
                                 description="Change the device used by BERT",
                                 valid_values=None),
    "tf": Configuration(config="tf",
                        description="Method used for Term Frequency calculation",
                        valid_values=['binary', 'raw', 'freq', 'log_norm', 'double_norm']),
    "idf": Configuration(config="idf",
                         description="Method used for Inverse Document Frequency calcualtion.",
                         valid_values=['smooth', 'probabilistic'])

}

configuration = dict(idf="smooth", tokenizer="bert", tf="raw", bert_device="cpu")


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
            if cfg.config == 'tf':
                if value not in cfg.valid_values:
                    raise Exception(
                        f"{value} is not a valid value for {config}. Choose one of {', '.join(cfg.valid_values)}")

            elif cfg.config == 'idf':
                if value not in cfg.valid_values:
                    raise Exception(
                        f"{value} is not a valid value for {config}. Choose one of {', '.join(cfg.valid_values)}")

            # Normalize User Inputs Based on Config Name
            elif cfg.config == 'word_tokenizer':
                value = value.lower().replace(' ', '').replace('-', '').replace('tokenizer', '')

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
    configuration[config] = value


@contextmanager
def tokenizer_context(tokenizer_name, warning=False):
    from .bblock import DocBuilder  # pylint: disable=import-outside-toplevel

    if warning:
        warnings.warn(f"Changing tokenizer to {tokenizer_name}")

    yield DocBuilder(tokenizer_name)


@contextmanager
def idf_context(idf_type, warning=False):
    current = configuration['idf']

    if warning and current != idf_type:
        warnings.warn(f"Changing idf function to {idf_type}")

    try:
        set_config('idf', idf_type)
        yield
    finally:
        set_config('idf', current)


@contextmanager
def tf_context(tf_type, warning=False):
    current = configuration['tf']

    if warning and current != tf_type:
        warnings.warn(f"Changing tf function to {tf_type}")

    try:
        set_config('tf', tf_type)
        yield
    finally:
        set_config('tf', current)


@contextmanager
def bert_device_context(device, warning=False):
    current = configuration["bert_device"]
    if warning and current != device:
        warnings.warn(f"Changing device for BERT to {device}")

    try:
        set_config('bert_device', device)
        yield
    finally:
        set_config('bert_device', current)

@check_config
def get_config(config: str):  # pylint: disable=inconsistent-return-statements
    return configuration[config]


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
