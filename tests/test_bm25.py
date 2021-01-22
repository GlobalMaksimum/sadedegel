import numpy as np
import pytest
from .context import Doc, load_raw_corpus

__famous_quote__ = "Merhaba dünya. Biz dostuz. Barış için geldik."


def test_bm25():
    d = Doc(__famous_quote__)
    assert np.sum(d[0].bm25()) == pytest.approx(13.99788)
    with pytest.raises(UserWarning, match=r"Out of empirical bounds *."):
        d[0].bm25(k1=0)


def test_bm25_on_corpus():
    raw = load_raw_corpus()
    for text in raw:
        d = Doc(text)
        for sent in d:
            assert isinstance(sent.bm25(), np.float32)
