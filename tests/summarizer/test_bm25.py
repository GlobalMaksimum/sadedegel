from .context import BM25Summarizer, Doc, tf_context, load_raw_corpus
import numpy as np
import pytest


@pytest.mark.skip()
@pytest.mark.parametrize("tf_type, result", [('binary', np.array([0.636, 0.364])),
                                             ('raw', np.array([0.636, 0.364])),
                                             ('freq', np.array([0.636, 0.3638])),
                                             ('log_norm', np.array([0.6357, 0.3638])),
                                             ('double_norm', np.array([0.636, 0.364]))])
def test_bm25(tf_type, result):
    with tf_context(tf_type) as Doc2:
        d = Doc2("Onu hiç sevmedim. Bu iş çok zor.")
        assert BM25Summarizer().predict(d) == pytest.approx(result)


def test_bm25_corpus():
    raw = load_raw_corpus()
    for text in raw:
        d = Doc(text)
        assert np.sum(BM25Summarizer().predict(d)) == pytest.approx(1)
