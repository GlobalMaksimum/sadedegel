import pytest

from .context import ICUTokenizerHelper, tokenizer_context, WordTokenizer


@pytest.mark.parametrize("text, tokens_true", [("Havaalanında bekliyoruz.", ['Havaalanında', 'bekliyoruz', '.']),
                                               ('Ali topu tut.', ['Ali', 'topu', 'tut', '.'])])
def test_icu_helper_tokenizer(text, tokens_true):
    st = ICUTokenizerHelper()
    tokens_pred = st(text)
    assert tokens_pred == tokens_true


def test_icu_tokenization_sentences():
    text = 'Merhaba dünya. Barış için geldik.'

    with tokenizer_context("icu") as Doc:
        doc = Doc(text)

        assert doc[0].tokens == ['Merhaba', 'dünya', '.']
        assert doc[1].tokens == ['Barış', 'için', 'geldik', '.']
