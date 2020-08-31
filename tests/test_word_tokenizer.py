import pytest
from .context import Doc, SimpleTokenizer, BertTokenizer, tokenizer_context, WordTokenizer


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

        assert doc.sents[0].tokens == ['Merhaba', 'dünya', '.']
        assert doc.sents[1].tokens == ['Barış', 'için', 'geldik', '.']


def test_bert_tokenization_sents():
    text = 'Merhaba dünya. Barış için geldik.'

    with tokenizer_context(BertTokenizer.__name__):
        doc = Doc(text)

        assert doc.sents[0].tokens == ['Merhaba', 'dünya', '.']
        assert doc.sents[1].tokens == ['Barış', 'için', 'geldik', '.']


def test_singleton_tokenizer():
    st1 = WordTokenizer.factory('simple')
    st2 = WordTokenizer.factory('simple-tokenizer')
    st3 = WordTokenizer.factory('SimpleTokenizer')

    assert st1 == st2 == st3

    bt1 = WordTokenizer.factory('bert')
    bt2 = WordTokenizer.factory('bert-tokenizer')
    bt3 = WordTokenizer.factory('BERTTokenizer')

    assert bt1 == bt2 == bt3
