from pytest import approx
import numpy as np
from tests.context import RandomSummarizer, PositionSummarizer, LengthSummarizer, BandSummarizer, Rouge1Summarizer
from tests.context import Doc


def list_equal(a: list, b: list) -> bool:
    return all((_a == _b) for _a, _b in zip(a, b))


def test_first_default():
    summarizer = PositionSummarizer(normalize=False)

    assert list_equal(summarizer.predict([0, 1, 2]), [2, 1, 0])


def test_first_normalized():
    summarizer = PositionSummarizer()

    assert summarizer.predict([0, 1, 2]) == approx([2 / 3, 1 / 3, 0])


def test_random_default():
    summarizer = RandomSummarizer(normalize=False)

    assert summarizer.predict([0, 1]) == approx(np.array([0.37454012, 0.95071431]))


def test_random_normalized():
    summarizer = RandomSummarizer(normalize=True)

    assert summarizer.predict([0, 1]) == approx(np.array([0.28261752, 0.71738248]))


def test_token_length_summarizer():
    summarizer = LengthSummarizer(normalize=False)

    assert summarizer.predict(
        Doc('Meksikalılaştıramadıklarımızdan olduğunuz kesin. Meksikalı olmadığımızın bilincindeyim.').sents) == approx(
        np.array([9, 7]))


def test_token_length_summarizer_normalized():
    summarizer = LengthSummarizer(normalize=True)

    assert summarizer.predict(
        Doc('Meksikalılaştıramadıklarımızdan olduğunuz kesin. Meksikalı olmadığımızın bilincindeyim.').sents) == approx(
        np.array([9 / 16, 7 / 16]))


def test_char_length_summarizer():
    summarizer = LengthSummarizer(mode="char", normalize=False)

    assert summarizer.predict(
        Doc('Meksikalılaştıramadıklarımızdan olduğunuz kesin. Meksikalı olmadığımızın bilincindeyim.').sents) == approx(
        np.array([56, 42]))


def test_token_char_summarizer_normalized():
    summarizer = LengthSummarizer(mode="char", normalize=True)

    assert summarizer.predict(
        Doc('Meksikalılaştıramadıklarımızdan olduğunuz kesin. Meksikalı olmadığımızın bilincindeyim.').sents) == approx(
        np.array([56 / 98, 42 / 98]))
