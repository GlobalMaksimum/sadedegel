import glob
from os.path import dirname, join
from loguru import logger
import json
import re

from .util import safe_json_load, safe_read



def file_paths():
    base_path = dirname(__file__)

    search_pattern = join(base_path, 'raw', '*.txt')

    files = sorted(glob.glob(search_pattern))

    return files


def cleaner(doc: str):
    return re.sub('\d+\s+[a-zA-ZŞşğĞüÜıİ]+\s+\d{4}\s+PAYLAŞ\s+yorum\s+yaz(\s+a)?', '', doc, flags=re.I)


def load_raw_corpus(return_iter: bool = True, base_path=None, clean=True):
    """Load corpus of sample news tokenized into sentences.

    Examples
    --------
    >>> from sadedegel.dataset import load_raw_corpus
    >>> sents = load_raw_corpus(return_iter=False)
    >>> type(sents[0])
    <class 'str'>

    """
    if base_path is None:
        base_path = dirname(__file__)

    search_pattern = join(base_path, 'raw', '*.txt')

    logger.debug("Search path {}".format(search_pattern))

    files = sorted(glob.glob(search_pattern))

    if return_iter:
        if clean:
            return (cleaner(safe_read(file)) for file in files)
        else:
            return (safe_read(file) for file in files)
    else:
        if clean:
            return [cleaner(safe_read(file)) for file in files]
        else:
            return [safe_read(file) for file in files]


def load_sentence_corpus(return_iter: bool = True, base_path=None):
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
    if base_path is None:
        base_path = dirname(__file__)

    search_pattern = join(base_path, 'sents', '*.json')

    logger.debug("Search path {}".format(search_pattern))

    files = sorted(glob.glob(search_pattern))

    if return_iter:
        return map(safe_json_load, files)
    else:
        return [safe_json_load(file) for file in files]
