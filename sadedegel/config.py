__all__ = ['set_config', 'get_config', 'describe_config', 'get_all_configs', 'tokenizer_context']

from typing import Any
from collections import defaultdict
from contextlib import contextmanager
from configparser import ConfigParser
from pathlib import Path
from os.path import dirname
import warnings
from rich.console import Console
from rich.table import Table

from .about import __version__


def set_config(config: str, value: Any):  # pylint: disable=unused-argument
    if tuple(map(int, __version__.split('.'))) < (0, 18):  # pylint: disable=no-else-raise
        raise DeprecationWarning(
            "set_config is deprecated with 0.16. Use *_context functions for runtime configuration changes.")
    else:
        raise Exception("set_config should be removed.")


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


def get_config(config: str):  # pylint: disable=unused-argument
    if tuple(map(int, __version__.split('.'))) < (0, 18):  # pylint: disable=no-else-raise
        raise DeprecationWarning(
            "get_config is deprecated with 0.16. Use `sadedegel config` command to retrieve configuration")
    else:
        raise Exception("get_config function should be removed.")


def describe_config(config: str, print_desc=False):  # pylint: disable=unused-argument
    if tuple(map(int, __version__.split('.'))) < (0, 18):  # pylint: disable=no-else-raise
        raise DeprecationWarning(
            "get_config is deprecated with 0.16. Use `sadedegel config` command to retrieve configuration")
    else:
        raise Exception("describe_config should be removed.")


def get_all_configs():
    if tuple(map(int, __version__.split('.'))) < (0, 18):  # pylint: disable=no-else-raise
        raise DeprecationWarning(
            "get_config is deprecated with 0.16. Use `sadedegel config` command to retrieve configuration")
    else:
        raise Exception("describe_config should be removed.")


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
                    "tf__method": "Method used in term frequency calculation",
                    "tf__double_norm_k": "Smooth parameter used by double norm term frequency method.",
                    "idf__method": "Method used in Inverse Document Frequency calculation"}

    default_config = ConfigParser()
    default_config.read([Path(dirname(__file__)) / 'default.ini'])

    console = Console()

    table = Table(show_header=True, header_style="bold #2070b2")

    table.add_column("section")
    table.add_column("parameter_name")
    table.add_column("current_value")
    table.add_column("default_value")
    table.add_column("description", width=40)

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
