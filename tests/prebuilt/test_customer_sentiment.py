from .context import customer_sentiment
from sklearn.linear_model import SGDClassifier


def test_model_load():
    model = customer_sentiment.load('v2')
    assert isinstance(model['sgd_model'], SGDClassifier)


def test_inference():
    model = customer_sentiment.load('v2')
    pred = model.predict(['ürün şahane.'])
    pred_sentiment = customer_sentiment._classes[pred[0]]
    assert pred[0] == 1
    assert pred_sentiment == 'POSITIVE'
