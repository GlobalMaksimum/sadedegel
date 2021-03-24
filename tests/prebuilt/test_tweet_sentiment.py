import pytest
from .context import tweet_sentiment


@pytest.mark.skip()
def test_model_load():
    model = tweet_sentiment.load()
    pred = model.predict(['süper bir haber bu'])

    assert pred[0] == 0
