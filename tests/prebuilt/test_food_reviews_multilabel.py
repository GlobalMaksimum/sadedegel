from .context import food_reviews_multilabel

def test_model_load():
    model = food_reviews_multilabel.load()
    pred_neg = model.predict(['on numara ama biraz yavaştı'])
    pred_pos = model.predict(['sıcacık geldi ve lezzetliydi'])

    assert all(a == b for a, b in zip([0, 1, 1], pred_neg[0].tolist()))
    assert all([a == b for a, b in zip([1, 1, 1], pred_pos[0].tolist())])
