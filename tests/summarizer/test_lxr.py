from pytest import approx
import numpy as np
from tests.context import LexRankSummarizer
from sadedegel.tokenize import Doc

def test_lxr_summarizer_all_lower():
    summ = LexRankSummarizer(normalize=False)

    assert summ.predict(Doc('ali topu tut. oya ip atla. ahmet topu at.').sents) == approx(np.array([1.,1.,1.]))


def test_lxr_summarizer_proper_case():
    summ = LexRankSummarizer(normalize=False)
    assert summ.predict(Doc('Ali topu tut. Oya ip atla. Ahmet topu at.').sents) == approx(np.array([1.,1.,1.]))

def test_lxr_summarize_text():
    summ = LexRankSummarizer()
    doc = Doc('ali topu tut. oya ip atla. ahmet topu at.')

    assert summ(doc, k=1) == [doc.sents[2]]
