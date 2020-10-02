import pytest
from .context import Token, Doc


famous_quote = "Merhaba dünya. 36 ışıkyılı uzaktan geldik."


def test_token():

    d = Doc(famous_quote)
    s = d.sents[0]
    t = s.tokens[0]

    assert isinstance(t) == str


def test_is_digit():

    t = Token(0, '123', 10, 100)
    assert t.is_digit


def test_is_punct():

    t = Token(0, '!', 10, 100)
    assert t.is_punct


@pytest.mark.parametrize("word, shape", [('1969', 'dddd'),
                                         ('Hello', 'Xxxxx')])
def test_shape(word, shape):
    t = Token(0, word, 10, 100)

    assert t.shape == shape
