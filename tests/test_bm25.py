import pkgutil  # noqa: F401 # pylint: disable=unused-import

from itertools import product
import numpy as np
import pytest
from .context import Doc, load_raw_corpus, tokenizer_context

__famous_quote__ = "Merhaba dünya. Biz dostuz. Barış için geldik."


@pytest.mark.skip()
def test_bm25():
    d = Doc(__famous_quote__)
    assert np.sum(d[0].bm25) == pytest.approx(13.99788 * len(d[0].tokens) / 18.14)
    with pytest.raises(UserWarning, match=r"Out of empirical bounds *."):
        d[0].bm25(k1=0)


def test_bm25_type_sanity():
    raw = load_raw_corpus()
    for text in raw:
        d = Doc(text)
        for sent in d:
            assert isinstance(sent.bm25, np.float32)


tfs = ["binary", "raw", "freq", "log_norm", "double_norm"]
idfs = ["smooth", "probabilistic"]


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
@pytest.mark.parametrize("tf_type, idf_type", product(tfs, idfs))
def test_get_bm25(tf_type, idf_type):
    with tokenizer_context("bert") as cDoc:
        raw = load_raw_corpus()
        for text in raw:
            d = cDoc(text)
            for sent in d:
                assert len(sent.get_bm25(tf_type, idf_type, 1.25, 0.75)) == 27403
