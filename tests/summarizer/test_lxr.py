from pytest import approx
import pytest
import numpy as np
from tests.context import LexRankSummarizer
from sadedegel import Doc


@pytest.mark.skip()
def test_lxr_summarizer_all_lower():
    summ = LexRankSummarizer("log_norm", "smooth", normalize=False)

    assert summ.predict(Doc('ali topu tut. oya ip atla. ahmet topu at.')) == approx(np.array([1., 1., 1.]))


@pytest.mark.skip()
def test_lxr_summarizer_proper_case():
    summ = LexRankSummarizer("log_norm", "smooth", normalize=False)
    assert summ.predict(Doc('Ali topu tut. Oya ip atla. Ahmet topu at.')) == approx(np.array([1., 1., 1.]))


def test_lxr_summarize_text():
    summ = LexRankSummarizer("log_norm", "smooth")
    doc = Doc('ali topu tut. oya ip atla. ahmet topu at.')

    assert summ(doc, k=1) == [doc[2]]
