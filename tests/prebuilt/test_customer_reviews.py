from .context import customer_reviews_classification, CLASS_VALUES_CUST

def test_model_load():
    model = customer_reviews_classification.load()
    pred_hotel = model.predict(['odalar çok kirliydi'])
    pred_pc = model.predict(['ram çok düşük'])

    assert CLASS_VALUES_CUST[pred_hotel[0]] == 'turizm'
    assert CLASS_VALUES_CUST[pred_pc[0]] == 'bilgisayar'

    prob_hotel = model.predict_proba(['odalar çok kirliydi'])

    assert prob_hotel[0][30] == prob_hotel[0].max()
    assert prob_hotel.shape == (1, 32)