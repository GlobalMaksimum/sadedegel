import pytest
from .context import news_classification, load_raw_corpus


@pytest.mark.skip()
def test_classify():
    raw = load_raw_corpus()
    model = news_classification.load()

    y_pred = model.predict(raw)

    assert len(y_pred) == 98
