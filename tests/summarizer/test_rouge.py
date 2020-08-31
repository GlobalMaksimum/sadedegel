from pytest import approx, raises
import numpy as np
import pytest
from .context import Rouge1Summarizer, Doc, tokenizer_context, SimpleTokenizer, BertTokenizer

tokenizer_parameter = [pytest.param(SimpleTokenizer.__name__, marks=pytest.mark.xfail), BertTokenizer.__name__]


@pytest.mark.parametrize("tokenizer", tokenizer_parameter)
def test_rouge1_summarizer_precision_all_lower(tokenizer):
    with tokenizer_context(tokenizer):
        summ = Rouge1Summarizer(normalize=False, metric="precision")
        assert summ.predict(Doc('ali topu tut. oya ip atla. ahmet topu at.').sents) == approx(
            np.array([0.5, 0.4, 0.75]))


@pytest.mark.parametrize("tokenizer", tokenizer_parameter)
def test_rouge1_summarizer_precision_proper_case(tokenizer):
    with tokenizer_context(tokenizer):
        summ = Rouge1Summarizer(normalize=False, metric="precision")

        assert summ.predict(Doc('Ali topu tut. Oya ip atla. Ahmet topu at.').sents) == approx(
            np.array([0.5, 0.4, 0.75]))


@pytest.mark.parametrize("tokenizer", tokenizer_parameter)
def test_rouge1_summarizer_recall_all_lower(tokenizer):
    with tokenizer_context(tokenizer):
        summ = Rouge1Summarizer(normalize=False, metric="recall")

        assert summ.predict(Doc('ali topu tut. oya ip atla. ahmet topu at.').sents) == approx(
            np.array([2 / 9, 2 / 8, 3 / 9]))


@pytest.mark.parametrize("tokenizer", tokenizer_parameter)
def test_rouge1_summarizer_recall_proper_case(tokenizer):
    with tokenizer_context(tokenizer):
        summ = Rouge1Summarizer(normalize=False, metric="recall")
        assert summ.predict(Doc('Ali topu tut. Oya ip atla. Ahmet topu at.').sents) == approx(
            np.array([2 / 9, 2 / 8, 3 / 9]))


@pytest.mark.parametrize("tokenizer", tokenizer_parameter)
def test_rouge1_summarizer_f1_all_lower(tokenizer):
    with tokenizer_context(tokenizer):
        summ = Rouge1Summarizer(normalize=False)
        assert summ.predict(Doc('ali topu tut. oya ip atla. ahmet topu at.').sents) == approx(
            np.array([0.30769231, 0.30769231, 0.46153846]))


@pytest.mark.parametrize("tokenizer", tokenizer_parameter)
def test_rouge1_summarizer_f1_proper_case(tokenizer):
    with tokenizer_context(tokenizer):
        summ = Rouge1Summarizer(normalize=False)
        assert summ.predict(Doc('Ali topu tut. Oya ip atla. Ahmet topu at.').sents) == approx(
            np.array([0.30769231, 0.30769231, 0.46153846]))


@pytest.mark.parametrize("tokenizer", tokenizer_parameter)
def test_rouge1_summarize_text(tokenizer):
    with tokenizer_context(tokenizer):
        summ = Rouge1Summarizer()
        doc = Doc('ali topu tut. oya ip atla. ahmet topu at.')

        assert summ(doc, k=1) == [doc.sents[2]]


def test_rouge1_summarizer_unknown_mode():
    with raises(ValueError):
        _ = Rouge1Summarizer('unknown')
