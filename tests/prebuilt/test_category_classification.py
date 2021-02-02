from .context import news_classification, load_raw_corpus


def test_classify():
    raw = load_raw_corpus()
    model = news_classification.load()

    y_pred = model.predict(raw)

    assert len(y_pred) == 98
