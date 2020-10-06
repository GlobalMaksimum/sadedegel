from .context import TFIDFSummarizer, Doc, tf_context
import numpy as np
import pytest


@pytest.mark.parametrize("tf_type, result", [('binary', np.array([0.63611803, 0.36388197])),
                                             ('raw', np.array([0.62492017, 0.37507983])),
                                             ('freq', np.array([0.62492017, 0.37507983])),
                                             ('log_norm', np.array([0.62933615, 0.37066385])),
                                             ('double_norm', np.array([0.63216882, 0.36783118]))])
def test_tfidf(tf_type, result):
    with tf_context(tf_type):
        d = Doc("Onu hiç sevmedim. Bu iş çok zor.")
        assert TFIDFSummarizer().predict(d.sents) == pytest.approx(result)
