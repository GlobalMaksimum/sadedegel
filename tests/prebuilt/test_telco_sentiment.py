from .context import telco_sentiment, SENTIMENT_VALUES_T


def test_model_load():
    pipeline = telco_sentiment.load()
    assert True


def test_inference():
    model = telco_sentiment.load()

    pred = model.predict(['turkcell en iyi operatör.', 'burada hala çekmiyor amk.'])

    assert SENTIMENT_VALUES_T[pred[0]] in SENTIMENT_VALUES_T
    assert SENTIMENT_VALUES_T[pred[1]] in SENTIMENT_VALUES_T

    probability = model.predict_proba(['turkcell en iyi operatör.', 'burada hala çekmiyor amk.'])

    assert probability.shape == (2, 3)
