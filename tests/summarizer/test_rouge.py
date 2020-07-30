from pytest import approx
import numpy as np
from tests.context import Rouge1Summarizer
from sadedegel.tokenize import Doc


def test_rouge1_summarizer_precision_all_lower():
    summ = Rouge1Summarizer(normalize=False, metric="precision")

    assert summ.predict(Doc('ali topu tut. oya ip atla. ahmet topu at.').sents) == approx(np.array([0.5, 0.4, 0.75]))


def test_rouge1_summarizer_precision_proper_case():
    summ = Rouge1Summarizer(normalize=False, metric="precision")

    assert summ.predict(Doc('Ali topu tut. Oya ip atla. Ahmet topu at.').sents) == approx(np.array([0.5, 0.4, 0.75]))


def test_rouge1_summarizer_recall_all_lower():
    summ = Rouge1Summarizer(normalize=False, metric="recall")

    assert summ.predict(Doc('ali topu tut. oya ip atla. ahmet topu at.').sents) == approx(
        np.array([2 / 9, 2 / 8, 3 / 9]))


def test_rouge1_summarizer_recall_proper_case():
    summ = Rouge1Summarizer(normalize=False, metric="recall")

    assert summ.predict(Doc('Ali topu tut. Oya ip atla. Ahmet topu at.').sents) == approx(
        np.array([2 / 9, 2 / 8, 3 / 9]))


def test_rouge1_summarizer_f1_all_lower():
    summ = Rouge1Summarizer(normalize=False)

    assert summ.predict(Doc('ali topu tut. oya ip atla. ahmet topu at.').sents) == approx(
        np.array([0.30769231, 0.30769231, 0.46153846]))


def test_rouge1_summarizer_f1_proper_case():
    summ = Rouge1Summarizer(normalize=False)

    assert summ.predict(Doc('Ali topu tut. Oya ip atla. Ahmet topu at.').sents) == approx(
        np.array([0.30769231, 0.30769231, 0.46153846]))


def test_rouge1_summarize_text():
    summ = Rouge1Summarizer()
    doc = Doc('ali topu tut. oya ip atla. ahmet topu at.')

    assert summ(doc, k=1) == [doc.sents[2]]
