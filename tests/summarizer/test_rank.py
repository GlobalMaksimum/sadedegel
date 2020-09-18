import pytest
from pytest import warns, raises
from .context import TextRank, Doc
import numpy as np

famous_quote = ("Merhaba dünya biz dostuz. Barış için geldik. Sizi lazerlerimizle buharlaştırmayacağız."
                "Onun yerine kölemiz olacaksınız.")


def is_sorted(listable):
    l = list(listable)
    return all(a >= b for a, b in zip(l, l[1:]))


@pytest.mark.parametrize("input_type, normalize, text",
                         [pytest.param('bert', True, famous_quote, id='Normalized score'),
                          pytest.param('tfidf', True, famous_quote, id='TFIDF Input Type'),
                          pytest.param('bert', False, famous_quote, id='Raw score'),
                          pytest.param('bert', True, [], id='Empty Sentence List')])
def test_text_rank_sanity(input_type, normalize, text):
    if not text:
        with raises(ValueError, match=r"Ensure that document .*"):
            TextRank(input_type).predict(text)
    else:
        d = Doc(text)
        if input_type != 'bert':
            with raises(ValueError, match=r"mode should be one of .*"):
                TextRank(input_type)(d, 1)


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("text", [famous_quote])
def test_text_rank_descending(normalize, text):
    d = Doc(text)
    with warns(UserWarning, match="Changing tokenizer to"):
        scores = TextRank(alpha=0.5, normalize=normalize).predict(d)

        assert is_sorted(scores)


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("text", [famous_quote])
def test_text_rank_correct_number_of_sentences(normalize, text):
    d = Doc(text)
    with warns(UserWarning, match="Changing tokenizer to"):
        assert len(TextRank()(d, k=1)) == 1
