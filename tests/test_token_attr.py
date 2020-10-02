import pytest
from .context import Token

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
