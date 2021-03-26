import pytest
from .context import tweet_sentiment, SENTIMENT_VALUES


def test_model_load():
    model = tweet_sentiment.load()
    pred = model.predict(['harika bir haber bu'])

    assert SENTIMENT_VALUES[pred[0]] == 'POSITIVE'

    probability = model.predict_proba(['harika bir haber bu'])

    assert probability.shape == (1, 2)
