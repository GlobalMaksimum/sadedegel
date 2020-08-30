from .context import Doc, SimpleTokenizer, BertTokenizer, set_config


def test_bert_tokenizer_init():
    bt = BertTokenizer()
    assert str(bt) == 'BERT Tokenizer'


def test_simple_tokenizer_init():
    st = SimpleTokenizer()
    assert str(st) == 'Simple Tokenizer'


def test_bert_tokenizer():
    bt = BertTokenizer()
    text = "Havaalanında bekliyoruz."
    tokens, _ = bt.tokenize(text)
    assert tokens == ['[CLS]', 'Havaalanı', '##nda', 'bekliyoruz', '.', '[SEP]']


def test_simple_tokenizer():
    st = SimpleTokenizer()
    text = "Havaalanında bekliyoruz."
    tokens, _ = st.tokenize(text)
    assert tokens == ['havaalanında', 'bekliyoruz']


def test_simple_tokenization_sentences():
    text = 'Merhaba dünya. Barış için geldik.'

    set_config("word_tokenizer", SimpleTokenizer.name)
    doc = Doc(text)

    assert doc.sents[0].tokens == ['merhaba', 'dünya']
    assert doc.sents[1].tokens == ['barış', 'için', 'geldik']


def test_bert_tokenization_sents():
    text = 'Merhaba dünya. Barış için geldik.'
    set_config("word_tokenizer", BertTokenizer.name)
    doc = Doc(text)

    assert doc.sents[0].tokens == ['Merhaba', 'dünya', '.']
    assert doc.sents[1].tokens == ['Barış', 'için', 'geldik', '.']
