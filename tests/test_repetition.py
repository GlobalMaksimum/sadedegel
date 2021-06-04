import pytest

from .context import SimpleTokenizer, BertTokenizer, ICUTokenizer, Text2Doc


@pytest.mark.parametrize('toker, text, tokens_true', [
    (ICUTokenizer, 'alemiiiin kralı geliyooor geliyooooooor', ['alemin', 'kralı', 'geliyor', 'geliyor']),
    (SimpleTokenizer, 'alemiiiin kralı geliyooor geliyooooooor', ['alemin', 'kralı', 'geliyor', 'geliyor']),
    (BertTokenizer, 'alemiiiin kralı geliyooor geliyooooooor', ['alem', '##in', 'kralı', 'geliyor', 'geliyor'])
])
def test_tokenizer_emoji(text, tokens_true, toker):
    tokenizer = toker(repetition=True)
    tokens_pred = tokenizer(text)
    assert tokens_pred == tokens_true


@pytest.mark.parametrize('toker, text, tokens_true', [
    ('icu', ['alemiiiin kralı geliyooor geliyooooooor'], ['alemin', 'kralı', 'geliyor', 'geliyor']),
    ('simple', ['alemiiiin kralı geliyooor geliyooooooor'], ['alemin', 'kralı', 'geliyor', 'geliyor']),
    ('bert', ['alemiiiin kralı geliyooor geliyooooooor'], ['alem', '##in', 'kralı', 'geliyor', 'geliyor'])
])


def test_t2d_hashtag(text, tokens_true, toker):
    tokenizer = Text2Doc(tokenizer=toker, repetition=True)
    tokens_pred = tokenizer.transform(text)
    assert tokens_pred[0].tokens == tokens_true
