from .context import tweet_profanity
from sklearn.linear_model import SGDClassifier


def test_model_load():
    model = tweet_profanity.load('v1')
    assert isinstance(model['sgd_model'], SGDClassifier)


def test_inference():
    model = tweet_profanity.load('v1')
    pred = model.predict(['amk hepinizin.', 'harika fikir.'])
    assert tweet_profanity._classes[pred[0]] == 'PROFANE'
    assert tweet_profanity._classes[pred[1]] == 'PROPER'
