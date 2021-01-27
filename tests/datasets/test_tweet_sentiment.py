from .context import load_tweet_sentiment_train, CLASS_VALUES


def test_data_load():
    data = load_tweet_sentiment_train()
    for i, row in enumerate(data):
        assert any(key in row.keys() for key in ['text_uuid', 'text', 'sentiment'])
        assert isinstance(row['text_uuid'], str)
        assert isinstance(row['text'], str)
        assert CLASS_VALUES[row['sentiment']] in ['POSITIVE', 'NEGATIVE']
    assert i == 11116
