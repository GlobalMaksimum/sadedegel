from .context import tweet_sentiment
from sklearn.linear_model import SGDClassifier


def test_model_load():
    model = tweet_sentiment.load()
    pred = model.predict(['süper bir haber bu'])

    assert pred[0] == 0
