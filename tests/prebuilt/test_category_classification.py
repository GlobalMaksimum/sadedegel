import pytest
from .context import news_classification, load_raw_corpus


def test_classify():
    raw = load_raw_corpus(False)
    model = news_classification.load()

    y_pred = model.predict(raw)

    assert len(y_pred) == 98

    probability = model.predict_proba(raw)

    assert probability.shape == (98, 12)
