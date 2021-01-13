from .context import customer_sentiment
from sklearn.naive_bayes import MultinomialNB


def test_model_load():
    model = customer_sentiment.load('v1')
    assert isinstance(model['nb_model'], MultinomialNB)


def test_inference():
    model = customer_sentiment.load('v1')
    pred = model.predict(['ürün şahane.'])
    pred_sentiment = customer_sentiment._classes[pred[0]]
    assert pred[0] == 2
    assert pred_sentiment == 'POSITIVE'
