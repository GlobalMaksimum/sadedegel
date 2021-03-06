import pkgutil  # noqa: F401 # pylint: disable=unused-import

from math import ceil
import pytest
from pytest import raises
from .context import Token, config_context, idf_context

famous_quote = "Merhaba dünya. 36 ışıkyılı uzaktan geldik."


def test_is_digit():
    t = Token('123')
    assert t.is_digit


def test_is_punct():
    t = Token('!')
    assert t.is_punct


@pytest.mark.parametrize("word, shape", [('1969', 'dddd'),
                                         ('Hello', 'Xxxxx')])
def test_shape(word, shape):
    t = Token(word)

    assert t.shape == shape


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
@pytest.mark.parametrize('idf_type, idf', [('smooth', 6),
                                           ('probabilistic', 5)])
def test_idf(idf_type, idf):
    with config_context(idf__method=idf_type, tokenizer='bert') as Doc:
        word = Doc('merhaba')
        t = word[0][0]  # first token of first sentence.
        assert ceil(t.idf) == idf


def test_invalid_idf_method():
    with raises(Exception, match=r"Unknown term frequency method.*"):
        with idf_context("plain") as _:
            pass
