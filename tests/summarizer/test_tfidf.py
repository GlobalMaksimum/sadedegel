from .context import TFIDFSummarizer, Doc
import numpy as np
import pytest


def test_tfidf():
    d = Doc("Onu hiç sevmedim. Bu iş çok zor.")
    assert TFIDFSummarizer().predict(d.sents) == pytest.approx(np.array([0.63611803, 0.36388197]))
