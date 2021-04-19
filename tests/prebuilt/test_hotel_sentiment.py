import pytest
from .context import hotel_sentiment, CLASS_VALUES_HOTEL


def test_model_load():
    _ = hotel_sentiment.load()


def test_inference():
    model = hotel_sentiment.load()

    pred = model.predict(['asla gidilmeyecek bir otel hasta olduk otel tam anlamıyla bir fiyasko satın alırken',
                          'Çok güzeldi 4 cü gidişimiz tesise cok memnun kaldık.'])

    assert CLASS_VALUES_HOTEL[pred[0]] == 'NEGATIVE'
    assert CLASS_VALUES_HOTEL[pred[1]] == 'POSITIVE'

    probability = model.predict_proba(
        ['asla gidilmeyecek bir otel hasta olduk otel tam anlamıyla bir fiyasko satın alırken',
         'Çok güzeldi 4 cü gidişimiz tesise cok memnun kaldık.'])

    assert probability.shape == (2, 2)
