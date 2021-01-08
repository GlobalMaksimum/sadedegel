import pytest
from .context import __emojis__
from .context import Doc
from .context import tokenizer_context


__famous_quote__ = "Merhaba dünya 🌎 . Biz dostuz 👽 ."
__famous_quote2__ = "Merhaba dünya🌎. Biz dostuz👽."
__famous_quote3__ = "🤪🤪🤪🤪🤭🤭🤭🤭🤭"


def test_emoji_bank():
    assert len(__emojis__) == 3859


@pytest.mark.parametrize("tokenizer", ["simple", "bert"])
def test_emoji_tokenization(tokenizer):
    with tokenizer_context(tokenizer) as Doc2:
        d = Doc2(__famous_quote__)
        d2 = Doc2(__famous_quote2__)
        d3 = Doc2(__famous_quote3__)
    if tokenizer == 'simple':
        assert d.tokens == ["Merhaba", "dünya", "🌎", ".", "Biz", "dostuz", "👽", "."]
        assert d2.tokens == ["Merhaba", "dünya", "🌎", ".", "Biz", "dostuz", "👽", "."]
        assert d3.tokens == []
    elif tokenizer == 'bert':
        assert d.tokens == ["Merhaba", "dünya", "🌎", ".", "Biz", "dostu", "##z", "👽", "."]
        assert d2.tokens == ["Merhaba", "dünya", "🌎", ".", "Biz", "dostu", "##z", "👽", "."]
        assert d3.tokens == ["🤪", "🤪", "🤪", "🤪", "🤭", "🤭", "🤭", "🤭", "🤭"]


def test_emoji_vectorization():
    d = Doc(__famous_quote__)
    d2 = Doc(__famous_quote3__)

    assert d.bert_embeddings.shape == (2, 768)
    assert d.tfidf_embeddings.shape == (2, 27744)

    assert d2.bert_embeddings.shape == (1, 768)
    assert d2.tfidf_embeddings.shape == (1, 27744)
