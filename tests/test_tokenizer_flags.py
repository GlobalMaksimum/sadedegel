import pytest

from .context import SimpleTokenizer, BertTokenizer, ICUTokenizer, Text2Doc


@pytest.mark.parametrize('toker, text, tokens_true', [
    (ICUTokenizer, '👍basarili ve kaliteli bir urun .', ['👍', 'basarili', 've', 'kaliteli', 'bir', 'urun', '.']),
    (SimpleTokenizer, '👍basarili ve kaliteli bir urun .', ['👍', 'basarili', 've', 'kaliteli', 'bir', 'urun', '.']),
    (BertTokenizer, '👍basarili ve kaliteli bir urun .',
     ['👍', 'basar', '##ili', 've', 'kaliteli', 'bir', 'ur', '##un', '.'])
])
def test_tokenizer_emoji(text, tokens_true, toker):
    tokenizer = toker(emoji=True)
    tokens_pred = tokenizer(text)
    assert tokens_pred == tokens_true


@pytest.mark.parametrize('toker, text, tokens_true', [
    (ICUTokenizer, 'çok güzel kütüphane olmuş @sadedegel', ['çok', 'güzel', 'kütüphane', 'olmuş', '@sadedegel']),
    (SimpleTokenizer, 'çok güzel kütüphane olmuş @sadedegel', ['çok', 'güzel', 'kütüphane', 'olmuş', '@sadedegel']),
    (BertTokenizer, 'çok güzel kütüphane olmuş @sadedegel', ['çok', 'güzel', 'kütüphane', 'olmuş', '@sadedegel'])
])
def test_tokenizer_mention(text, tokens_true, toker):
    tokenizer = toker(mention=True)
    tokens_pred = tokenizer(text)
    assert tokens_pred == tokens_true


@pytest.mark.parametrize('toker, text, tokens_true', [
    (ICUTokenizer, 'ağaçlar yanmasın! #yeşiltürkiye', ['ağaçlar', 'yanmasın', '!', '#yeşiltürkiye']),
    (SimpleTokenizer, 'ağaçlar yanmasın! #yeşiltürkiye', ['ağaçlar', 'yanmasın', '#yeşiltürkiye']),
    (BertTokenizer, 'ağaçlar yanmasın! #yeşiltürkiye', ['ağaçlar', 'yanma', '##sın', '!', '#yeşiltürkiye'])
])
def test_tokenizer_hashtag(text, tokens_true, toker):
    tokenizer = toker(hashtag=True)
    tokens_pred = tokenizer(text)
    assert tokens_pred == tokens_true


# Text2Doc Tests

@pytest.mark.parametrize('toker, text, tokens_true', [
    ('icu', ['👍basarili ve kaliteli bir urun .'], ['👍', 'basarili', 've', 'kaliteli', 'bir', 'urun', '.']),
    ('simple', ['👍basarili ve kaliteli bir urun .'], ['👍', 'basarili', 've', 'kaliteli', 'bir', 'urun', '.']),
    ('bert', ['👍basarili ve kaliteli bir urun .'],
     ['👍', 'basar', '##ili', 've', 'kaliteli', 'bir', 'ur', '##un', '.'])
])
def test_t2d_emoji(text, tokens_true, toker):
    tokenizer = Text2Doc(tokenizer=toker, emoji=True)
    tokens_pred = tokenizer.transform(text)
    assert tokens_pred[0].tokens == tokens_true


@pytest.mark.parametrize('toker, text, tokens_true', [
    ('icu', ['çok güzel kütüphane olmuş @sadedegel'], ['çok', 'güzel', 'kütüphane', 'olmuş', '@sadedegel']),
    ('simple', ['çok güzel kütüphane olmuş @sadedegel'], ['çok', 'güzel', 'kütüphane', 'olmuş', '@sadedegel']),
    ('bert', ['çok güzel kütüphane olmuş @sadedegel'], ['çok', 'güzel', 'kütüphane', 'olmuş', '@sadedegel'])
])
def test_t2d_mention(text, tokens_true, toker):
    tokenizer = Text2Doc(tokenizer=toker, mention=True)
    tokens_pred = tokenizer.transform(text)
    assert tokens_pred[0].tokens == tokens_true


@pytest.mark.parametrize('toker, text, tokens_true', [
    ('icu', ['ağaçlar yanmasın! #yeşiltürkiye'], ['ağaçlar', 'yanmasın', '!', '#yeşiltürkiye']),
    ('simple', ['ağaçlar yanmasın! #yeşiltürkiye'], ['ağaçlar', 'yanmasın', '#yeşiltürkiye']),
    ('bert', ['ağaçlar yanmasın! #yeşiltürkiye'], ['ağaçlar', 'yanma', '##sın', '!', '#yeşiltürkiye'])
])
def test_t2d_hashtag(text, tokens_true, toker):
    tokenizer = Text2Doc(tokenizer=toker, hashtag=True)
    tokens_pred = tokenizer.transform(text)
    assert tokens_pred[0].tokens == tokens_true
