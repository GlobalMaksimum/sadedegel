from .context import movie_reviews, SENTIMENT_VALUES_M


def test_model_load():
    model = movie_reviews.load()
    pred_neg = model.predict(['çok sıkıcı bir filmdi'])
    pred_pos = model.predict(['süper aksiyon, tavsiye ederim'])

    assert SENTIMENT_VALUES_M[pred_pos[0]] == 'POSITIVE'
    assert SENTIMENT_VALUES_M[pred_neg[0]] == 'NEGATIVE'

    prob_pos = model.predict_proba(['süper aksiyon, tavsiye ederim'])

    assert prob_pos[0][1] >= 0.5
    assert prob_pos.shape == (1, 2)
