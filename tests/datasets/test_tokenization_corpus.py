import pytest
from pathlib import Path
from os.path import expanduser
from .context import tokenization, check_and_display


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
