import pkgutil  # noqa: F401 # pylint: disable=unused-import

import pytest
from .context import SimpleTokenizer, BertTokenizer, ICUTokenizer


@pytest.mark.parametrize('toker, text, tokens_true', [
    (ICUTokenizer, "komik:) :( ;) <3 :/ :p :P :d :D :-) :-( xd xD :)))) ^_^ bu da benimki 8=======D",
     ["komik", ":)", ":(", ";)", "<3", ":/", ":p", ":P", ":d", ":D", ":-)", ":-(", "xd", "xD", ":))))", "^_^", "bu",
      "da", "benimki", "8=======D"]),
    (SimpleTokenizer, "komik:) :( ;) <3 :/ :p :P :d :D :-) :-( xd xD :)))) ^_^",
     ["komik", ":)", ":(", ";)", "<3", ":/", ":p", ":P", ":d", ":D", ":-)", ":-(", "xd", "xD", ":))))", "^_^"]),
])
def test_tokenizer_emoji(text, tokens_true, toker):
    tokenizer = toker(emoticon=True)
    tokens_pred = tokenizer(text)
    assert tokens_pred == tokens_true


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
@pytest.mark.parametrize('toker, text, tokens_true', [
    (BertTokenizer, "komik:) :( ;) <3 :/ :p :P :d :D :-) :-( xd xD :)))) ^_^",
     ["komik", ":)", ":(", ";)", "<3", ":/", ":p", ":P", ":d", ":D", ":-)", ":-(", "xd", "xD", ":))))", "^_^"]),
])
def test_bert_tokenizer_emoji(text, tokens_true, toker):
    tokenizer = toker(emoticon=True)
    tokens_pred = tokenizer(text)
    assert tokens_pred == tokens_true


@pytest.mark.parametrize('toker, text, tokens_true', [
    (ICUTokenizer, "komik:)",
     ["komik", ":", ")"]),
    (SimpleTokenizer, "komik:)",
     ["komik"]),
])
def test_tokenizer_emoji_f(text, tokens_true, toker):
    tokenizer = toker(emoticon=False)
    tokens_pred = tokenizer(text)
    assert tokens_pred == tokens_true


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
@pytest.mark.parametrize('toker, text, tokens_true', [
    (BertTokenizer, "komik:) :( ;) <3 :/ :p :P :d :D :-) :-( xd xD :)))) ^_^",
     ["komik", ":)", ":(", ";)", "<3", ":/", ":p", ":P", ":d", ":D", ":-)", ":-(", "xd", "xD", ":))))", "^_^"]),
])
def test_bert_tokenizer_emoji_f(text, tokens_true, toker):
    tokenizer = toker(emoticon=False)
    tokens_pred = tokenizer(text)
    assert tokens_pred == tokens_true
