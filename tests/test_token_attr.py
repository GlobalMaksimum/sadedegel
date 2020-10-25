from math import ceil
import pytest
from pytest import raises
from .context import Token, idf_context, set_config, BertTokenizer

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


@pytest.mark.parametrize('idf_type, idf', [('smooth', 6),
                                           ('probabilistic', 5)])
def test_idf(idf_type, idf):
    tokenizer = BertTokenizer()
    with idf_context(idf_type):
        t = tokenizer.vocabulary['merhaba']
        assert ceil(t.idf) == idf


def test_idf_setting():
    with raises(Exception, match=r".*is not a valid value.*"):
        set_config('idf', 'plain')
