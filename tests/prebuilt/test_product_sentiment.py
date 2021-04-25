from .context import product_sentiment
from sklearn.linear_model import SGDClassifier

def test_model_load():
    model = product_sentiment.load()
    pred = model.predict(['çok kötü bir kulaklık.'])

    assert pred[0] == 0