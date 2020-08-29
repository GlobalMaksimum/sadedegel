import glob
from os.path import dirname, join, expanduser, basename, splitext
from loguru import logger
import numpy as np
import re
from enum import Enum
from .util import safe_json_load, safe_read
from pathlib import Path


class CorpusTypeEnum(Enum):
    RAW = ('raw', '*.txt')
    SENTENCE = ('sents', '*.json')
    ANNOTATED = ('annotated', '*.json')

    def __init__(self, dir, pattern):
        self.dir = dir
        self.pattern = pattern


def file_paths(corpus_type: CorpusTypeEnum = CorpusTypeEnum.RAW, noext=False, use_basename=False, base_path=None):
    if base_path is None:
        base_path = dirname(__file__)

    if type(corpus_type) == CorpusTypeEnum:
        search_pattern = Path(base_path).expanduser() / corpus_type.dir

        logger.debug(f"Search pattern for {corpus_type}: {search_pattern}")
        files = search_pattern.glob(corpus_type.pattern)
    else:
        raise ValueError(f"Ensure that corpus_type is a one of raw, sentence or annotated")

    if use_basename:
        if noext:
            return sorted([splitext(basename(fn))[0] for fn in files])
        else:
            return sorted([basename(fn) for fn in files])
    else:
        return sorted(files)


def cleaner(doc: str):
    return re.sub(r'\d+\s+[a-zA-ZŞşğĞüÜıİ]+\s+\d{4}\s+PAYLAŞ\s+yorum\s+yaz(\s+a)?', '', doc, flags=re.I)


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

    search_pattern = join(expanduser(base_path), 'raw', '*.txt')

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

    search_pattern = join(expanduser(base_path), 'sents', '*.json')

    logger.debug("Search path {}".format(search_pattern))

    files = sorted(glob.glob(search_pattern))

    if return_iter:
        return map(safe_json_load, files)
    else:
        return [safe_json_load(file) for file in files]


def load_annotated_corpus(return_iter: bool = True, base_path=None):
    """Load corpus of sample news tokenized into sentences and scored based on human annotation"""

    files = file_paths(CorpusTypeEnum.ANNOTATED, base_path)

    def to_dict(d):
        return dict(sentences=[s['content'] for s in d['sentences']],
                    relevance=np.array([s['deletedInRound'] for s in d['sentences']]))

    if return_iter:
        return map(to_dict,
                   map(safe_json_load, files))
    else:
        return [to_dict(safe_json_load(file)) for file in files]
