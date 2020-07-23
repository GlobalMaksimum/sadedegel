import pytest
from .context import load, load_raw_corpus


@pytest.mark.skip()
def test_word_on_raw():
    raw = load_raw_corpus()

    sg = load()

    _ = [sg(doc) for doc in raw]
