import glob
from os.path import dirname, join
from loguru import logger
from typing import Iterator
import json


def _load(files: Iterator[str]):
    for file in files:
        with open(file) as fp:
            yield fp.read()


def load_raw_corpus(return_iter: bool = True):
    """Load corpus of sample news tokenized into sentences.

    Examples
    --------
    >>> from sadedegel.dataset import load_raw_corpus
    >>> sents = load_raw_corpus(return_iter=False)
    >>> type(sents[0])
    <class 'str'>

    """
    base_path = dirname(__file__)

    search_pattern = join(base_path, 'raw', '*.txt')

    logger.debug("Search path {}".format(search_pattern))

    files = glob.glob(search_pattern)

    if return_iter:
        return _load(files)
    else:
        return list(_load(files))


def load_sentence_corpus(return_iter: bool = True):
    """Load corpus of sample news tokenized into sentences.

    Examples
    --------
    >>> from sadedegel.dataset import load_sentence_corpus
    >>> sents = load_sentence_corpus(return_iter=False)
    >>> type(sents[0])
    <class 'dict'>

    >>> len(sents[0]['sentences'])
    62
    """
    base_path = dirname(__file__)

    search_pattern = join(base_path, 'sents', '*.json')

    logger.debug("Search path {}".format(search_pattern))

    files = glob.glob(search_pattern)

    if return_iter:
        return map(lambda buf: json.loads(buf), _load(files))
    else:
        return [json.loads(buf) for buf in _load(files)]
