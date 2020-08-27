from .context import Doc, SimpleTokenizer, BertTokenizer, set_config


def test_berttokenizer_init():
    bt = BertTokenizer()
    assert str(bt) == 'bert'


def test_simpletokenizer_init():
    st = SimpleTokenizer()
    assert str(st) == 'simple'


def test_berttokenizer():
    bt = BertTokenizer()
    text = "Havaalanında bekliyoruz."
    tokens, _ = bt.tokenize(text)
    assert tokens == ['[CLS]', 'Havaalanı', '##nda', 'bekliyoruz', '.', '[SEP]']


def test_simpletokenizer():
    st = SimpleTokenizer()
    text = "Havaalanında bekliyoruz."
    tokens, _ = st.tokenize(text)
    assert tokens == ['havaalanında', 'bekliyoruz']


def test_simple_tokenization_sents():
    text = 'Merhaba dünya. Barış için geldik.'
    set_config("simple")
    doc = Doc(text)

    assert doc.sents[0].tokens == ['merhaba', 'dünya']
    assert doc.sents[1].tokens == ['barış', 'için', 'geldik']


def test_bert_tokenization_sents():
    text = 'Merhaba dünya. Barış için geldik.'
    set_config("bert")
    doc = Doc(text)

    assert doc.sents[0].tokens == ['Merhaba', 'dünya', '.']
    assert doc.sents[1].tokens == ['Barış', 'için', 'geldik', '.']
