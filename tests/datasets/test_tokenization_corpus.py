import pytest
from pathlib import Path
from os.path import expanduser
import numpy as np
from .context import tokenization, check_and_display, tok_eval


def test_submodule_import():
    assert 'Raw' in tokenization.raw._desc
    assert 'Tokenized' in tokenization.tokenized._desc


@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data")).exists()')
def test_raw():
    raw = tokenization.raw.load_corpus()

    assert sum(1 for _ in raw) == 302936


@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data")).exists()')
def test_tokenized():
    tokenized = tokenization.tokenized.load_corpus()

    assert sum(1 for _ in tokenized) == 302936


@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data")).exists()')
def test_metadata():
    stats = check_and_display("~/.sadedegel_data")

    assert stats['byte']['raw'] * 1e6 >= 1 * 1024 * 1024
    assert stats['byte']['raw'] * 1e6 >= 1 * 1024 * 1024


corpus_parameters = [(tokenization.raw.load_corpus, True, 302936),
                     (tokenization.raw.load_corpus, False, 302936),
                     (tokenization.tokenized.load_corpus, True, 302936),
                     (tokenization.tokenized.load_corpus, True, 302936)]


@pytest.mark.parametrize("loader, return_iter, expected_count", corpus_parameters)
def test_corpus_size(loader, return_iter, expected_count):
    docs = loader(return_iter=return_iter)

    assert sum(1 for _ in docs) == expected_count

    if not return_iter:
        assert len(docs) == expected_count


def test_evaluator():
    raw_docs = tokenization.raw.load_corpus()
    tokenized_docs = tokenization.tokenized.load_corpus()
    eval_raw, eval_tokenized = [], []
    for raw_doc, tokenized_doc in zip(raw_docs, tokenized_docs):
        ix = raw_doc['index']
        if ix == 2:
            break
        eval_raw.append(raw_doc)
        eval_tokenized.append(tokenized_doc)
    table = tok_eval(eval_raw, eval_tokenized, ["simple", "bert"])

    assert np.float16(table[0][1]) == np.float16(0.8315)
    assert np.float16(table[1][1]) == np.float16(0.8739)
