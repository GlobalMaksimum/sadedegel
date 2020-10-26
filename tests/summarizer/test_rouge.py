from pytest import approx, raises
import numpy as np
import pytest
from .context import Rouge1Summarizer, tokenizer_context, SimpleTokenizer, BertTokenizer


@pytest.mark.parametrize("tokenizer, score_true",
                         [(SimpleTokenizer.__name__, np.array([2 / 4, 1 / 4, 2 / 4])),
                          (BertTokenizer.__name__, np.array([2 / 4, 2 / 5, 3 / 4]))])
def test_rouge1_summarizer_precision_all_lower(tokenizer, score_true):
    with tokenizer_context(tokenizer) as Doc2:
        summ = Rouge1Summarizer(normalize=False, metric="precision")
        assert summ.predict(Doc2('ali topu tut. oya ip atla. ahmet topu at.').sents) == approx(
            score_true)


@pytest.mark.parametrize("tokenizer, score_true",
                         [(SimpleTokenizer.__name__, np.array([2 / 4, 1 / 4, 2 / 4])),
                          (BertTokenizer.__name__, np.array([2 / 4, 2 / 5, 3 / 4]))])
def test_rouge1_summarizer_precision_proper_case(tokenizer, score_true):
    with tokenizer_context(tokenizer) as Doc2:
        summ = Rouge1Summarizer(normalize=False, metric="precision")

        assert summ.predict(Doc2('Ali topu tut. Oya ip atla. Ahmet topu at.').sents) == approx(
            score_true)


@pytest.mark.parametrize("tokenizer, score_true",
                         [(SimpleTokenizer.__name__, np.array([2 / 8, 1 / 8, 2 / 8])),
                          (BertTokenizer.__name__, np.array([2 / 9, 2 / 8, 3 / 9]))])
def test_rouge1_summarizer_recall_all_lower(tokenizer, score_true):
    with tokenizer_context(tokenizer) as Doc2:
        summ = Rouge1Summarizer(normalize=False, metric="recall")

        assert summ.predict(Doc2('ali topu tut. oya ip atla. ahmet topu at.').sents) == approx(
            score_true)


@pytest.mark.parametrize("tokenizer, score_true",
                         [(SimpleTokenizer.__name__, np.array([2 / 8, 1 / 8, 2 / 8])),
                          (BertTokenizer.__name__, np.array([2 / 9, 2 / 8, 3 / 9]))])
def test_rouge1_summarizer_recall_proper_case(tokenizer, score_true):
    with tokenizer_context(tokenizer) as Doc2:
        summ = Rouge1Summarizer(normalize=False, metric="recall")
        assert summ.predict(Doc2('Ali topu tut. Oya ip atla. Ahmet topu at.').sents) == approx(
            score_true)


@pytest.mark.parametrize("tokenizer, score_true",
                         [(SimpleTokenizer.__name__, np.array([0.33333333, 0.16666667, 0.33333333])),
                          (BertTokenizer.__name__, np.array([0.30769231, 0.30769231, 0.46153846]))])
def test_rouge1_summarizer_f1_all_lower(tokenizer, score_true):
    with tokenizer_context(tokenizer) as Doc2:
        summ = Rouge1Summarizer(normalize=False)
        assert summ.predict(Doc2('ali topu tut. oya ip atla. ahmet topu at.').sents) == approx(
            score_true)


@pytest.mark.parametrize("tokenizer, score_true",
                         [(SimpleTokenizer.__name__, np.array([0.33333333, 0.16666667, 0.33333333])),
                          (BertTokenizer.__name__, np.array([0.30769231, 0.30769231, 0.46153846]))])
def test_rouge1_summarizer_f1_proper_case(tokenizer, score_true):
    with tokenizer_context(tokenizer) as Doc2:
        summ = Rouge1Summarizer(normalize=False)
        assert summ.predict(Doc2('Ali topu tut. Oya ip atla. Ahmet topu at.').sents) == approx(
            score_true)


@pytest.mark.parametrize("tokenizer", [SimpleTokenizer.__name__, BertTokenizer.__name__])
def test_rouge1_summarize_text(tokenizer):
    with tokenizer_context(tokenizer) as Doc2:
        summ = Rouge1Summarizer()
        doc = Doc2('ali topu tut. oya ip atla. ahmet topu at.')

        assert summ(doc, k=1) == [doc.sents[2]]


def test_rouge1_summarizer_unknown_mode():
    with raises(ValueError):
        _ = Rouge1Summarizer('unknown')
