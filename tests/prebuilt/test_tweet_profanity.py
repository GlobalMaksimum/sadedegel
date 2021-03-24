import pytest
from .context import tweet_profanity, CLASS_VALUES


def test_model_load():
    _ = tweet_profanity.load()


@pytest.mark.skip()
def test_inference():
    model = tweet_profanity.load()

    pred = model.predict(['amk hepinizin.', 'harika fikir.'])

    assert CLASS_VALUES[pred[0]] == 'OFF'
    assert CLASS_VALUES[pred[1]] == 'NOT'
