import pytest

from .context import SimpleTokenizer, BertTokenizer, ICUTokenizer


@pytest.mark.parametrize('toker, text, tokens_true', [
    (ICUTokenizer, '👍basarili ve kaliteli bir urun .', ['👍', 'basarili', 've', 'kaliteli', 'bir', 'urun', '.']),
    (SimpleTokenizer, '👍basarili ve kaliteli bir urun .', ['👍', 'basarili', 've', 'kaliteli', 'bir', 'urun', '.']),
    (BertTokenizer, '👍basarili ve kaliteli bir urun .',
     ['👍', 'basar', '##ili', 've', 'kaliteli', 'bir', 'ur', '##un', '.'])
])
def test_tokenizer_emoji(text, tokens_true, toker):
    it = toker(emoji=True)
    tokens_pred = it._tokenize(text)
    assert tokens_pred == tokens_true


@pytest.mark.parametrize('toker, text, tokens_true', [
    (ICUTokenizer, 'çok güzel kütüphane olmuş @sadedegel', ['çok', 'güzel', 'kütüphane', 'olmuş', '@sadedegel']),
    (SimpleTokenizer, 'çok güzel kütüphane olmuş @sadedegel', ['çok', 'güzel', 'kütüphane', 'olmuş', '@sadedegel']),
    (BertTokenizer, 'çok güzel kütüphane olmuş @sadedegel', ['çok', 'güzel', 'kütüphane', 'olmuş', '@sadedegel'])
])
def test_tokenizer_mention(text, tokens_true, toker):
    it = toker(mention=True)
    tokens_pred = it._tokenize(text)
    assert tokens_pred == tokens_true


@pytest.mark.parametrize('toker, text, tokens_true', [
    (ICUTokenizer, 'ağaçlar yanmasın! #yeşiltürkiye', ['ağaçlar', 'yanmasın', '!', '#yeşiltürkiye']),
    (SimpleTokenizer, 'ağaçlar yanmasın! #yeşiltürkiye', ['ağaçlar', 'yanmasın', 'yeşiltürkiye']),
    (BertTokenizer, 'ağaçlar yanmasın! #yeşiltürkiye', ['ağaçlar', 'yanma', '##sın', '!', '#yeşiltürkiye'])
])

def test_tokenizer_hashtag(text, tokens_true, toker):
    it = toker(hashtag=True)
    tokens_pred = it._tokenize(text)
    assert tokens_pred == tokens_true
