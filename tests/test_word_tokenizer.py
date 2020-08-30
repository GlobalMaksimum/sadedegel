import pytest
from .context import Doc, SimpleTokenizer, BertTokenizer, tokenizer_context


def test_bert_tokenizer_init():
    bt = BertTokenizer()
    assert str(bt) == 'BERT Tokenizer'


def test_simple_tokenizer_init():
    st = SimpleTokenizer()
    assert str(st) == 'Simple Tokenizer'


def test_bert_tokenizer():
    bt = BertTokenizer()
    text = "Havaalanında bekliyoruz."
    tokens = bt.tokenize(text)
    assert tokens == ['Havaalanı', '##nda', 'bekliyoruz', '.']


@pytest.mark.parametrize("text, tokens_true", [("Havaalanında bekliyoruz.", ['Havaalanında', 'bekliyoruz', '.']),
                                               ('Ali topu tut.', ['Ali', 'topu', 'tut', '.'])])
def test_simple_tokenizer(text, tokens_true):
    st = SimpleTokenizer()
    tokens_pred = st.tokenize(text)
    assert tokens_pred == tokens_true


def test_simple_tokenization_sentences():
    text = 'Merhaba dünya. Barış için geldik.'

    with tokenizer_context(SimpleTokenizer.name):
        doc = Doc(text)

        assert doc.sents[0].tokens == ['Merhaba', 'dünya', '.']
        assert doc.sents[1].tokens == ['Barış', 'için', 'geldik', '.']


def test_bert_tokenization_sents():
    text = 'Merhaba dünya. Barış için geldik.'

    with tokenizer_context(BertTokenizer.name):
        doc = Doc(text)

        assert doc.sents[0].tokens == ['Merhaba', 'dünya', '.']
        assert doc.sents[1].tokens == ['Barış', 'için', 'geldik', '.']
