import pytest
from pathlib import Path  # pylint: disable=unused-import
from os.path import expanduser  # pylint: disable=unused-import
from .context import load_extended_metadata, load_extended_raw_corpus, load_extended_sents_corpus


@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data")).exists()')
def test_metadata():
    md = load_extended_metadata()

    assert md['byte']['sents'] >= 100 * 1024 * 1024
    assert md['count']['sents'] == 36131
    assert md['count']['raw'] == 36131


@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data")).exists()')
def test_raw():
    raw = load_extended_raw_corpus()

    assert sum(1 for _ in raw) == 36131


@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data")).exists()')
def test_sentences():
    sents = load_extended_sents_corpus()

    assert sum(1 for _ in sents) == 36131


@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data")).exists()')
def test_sentences():
    sents = load_extended_sents_corpus()

    assert all((('sentences' in doc and 'rouge1' in doc) for doc in sents))
