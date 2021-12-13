import pkgutil  # noqa: F401 # pylint: disable=unused-import

import numpy as np
import pytest

from .context import Doc, SimpleTokenizer, BertTokenizer, tokenizer_context, WordTokenizer, ICUTokenizer
from .context import load_raw_corpus


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
def test_bert_tokenizer():
    bt = BertTokenizer()
    text = "Havaalanında bekliyoruz."
    tokens = bt(text)
    assert tokens == ['Havaalanı', '##nda', 'bekliyoruz', '.']


@pytest.mark.parametrize("text, tokens_true", [("Havaalanında bekliyoruz.", ['Havaalanında', 'bekliyoruz', '.']),
                                               ('Ali topu tut.', ['Ali', 'topu', 'tut', '.'])])
def test_simple_tokenizer(text, tokens_true):
    st = SimpleTokenizer()
    tokens_pred = st(text)
    assert tokens_pred == tokens_true


def test_simple_tokenization_sentences():
    text = 'Merhaba dünya. Barış için geldik.'

    with tokenizer_context(SimpleTokenizer.__name__):
        doc = Doc(text)

        assert doc[0].tokens == ['Merhaba', 'dünya', '.']
        assert doc[1].tokens == ['Barış', 'için', 'geldik', '.']


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
def test_bert_tokenization_sents():
    text = 'Merhaba dünya. Barış için geldik.'

    with tokenizer_context(BertTokenizer.__name__):
        doc = Doc(text)

        assert doc[0].tokens == ['Merhaba', 'dünya', '.']
        assert doc[1].tokens == ['Barış', 'için', 'geldik', '.']


def test_tokenizer_type():
    st1 = WordTokenizer.factory('simple')
    st2 = WordTokenizer.factory('simple-tokenizer')
    st3 = WordTokenizer.factory('SimpleTokenizer')

    assert isinstance(st1, SimpleTokenizer) == isinstance(st2, SimpleTokenizer) == isinstance(st3, SimpleTokenizer)

    if pkgutil.find_loader("transformers") is not None:
        bt1 = WordTokenizer.factory('bert')
        bt2 = WordTokenizer.factory('bert-tokenizer')
        bt3 = WordTokenizer.factory('BERTTokenizer')

        assert isinstance(bt1, BertTokenizer) and isinstance(bt2, BertTokenizer) and isinstance(bt3, BertTokenizer)

    icut1 = WordTokenizer.factory('icu')
    icut2 = WordTokenizer.factory('icu-tokenizer')
    icut3 = WordTokenizer.factory('ICUTokenizer')

    assert isinstance(icut1, ICUTokenizer) == isinstance(icut2, ICUTokenizer) == isinstance(icut3, ICUTokenizer)


@pytest.mark.parametrize("toker", ["bert", "simple", "icu"])
def test_word_counting(toker):
    if pkgutil.find_loader("transformers") is not None or toker != "bert":
        with tokenizer_context(toker) as D:
            docs = [D(text) for text in load_raw_corpus()]

            if toker == "bert":
                assert np.array([len(d) for d in docs]).mean() == pytest.approx(42.397959)
                assert np.array([len(s) for d in docs for s in d]).mean() == pytest.approx(17.9253910)
            elif toker == 'simple':
                assert np.array([len(d) for d in docs]).mean() == pytest.approx(42.397959)
                assert np.array([len(s) for d in docs for s in d]).mean() == pytest.approx(12.4616125)
            else:
                assert np.array([len(d) for d in docs]).mean() == pytest.approx(42.397959)
                assert np.array([len(s) for d in docs for s in d]).mean() == pytest.approx(13.5150421)
