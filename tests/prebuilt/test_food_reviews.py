from .context import food_reviews, CLASS_VALUES_FOOD

def test_model_load():
    model = food_reviews.load()
    pred_neg = model.predict(['iki saatte geldi'])
    pred_pos = model.predict(['müdavimi olduk'])

    assert pred_neg == 0
    assert pred_pos == 1

    assert CLASS_VALUES_FOOD[pred_pos[0]] == 'POSITIVE'
    assert CLASS_VALUES_FOOD[pred_neg[0]] == 'NEGATIVE'
