import pkgutil  # noqa: F401 # pylint: disable=unused-import

import pytest
from pytest import approx, raises
import numpy as np
from .context import RandomSummarizer, PositionSummarizer, LengthSummarizer, BandSummarizer, tokenizer_context


def list_equal(a: list, b: list) -> bool:
    return all((_a == _b) for _a, _b in zip(a, b))


@pytest.mark.parametrize("input", [['0', '1', '2']])
def test_first_default(input):
    summarizer = PositionSummarizer(normalize=False)

    assert list_equal(summarizer.predict(input), [2, 1, 0])


def test_first_normalized():
    summarizer = PositionSummarizer()

    assert summarizer.predict(['0', '1', '2']) == approx([2 / 3, 1 / 3, 0])


@pytest.mark.parametrize("input", [['0', '1', '2']])
def test_last(input):
    summarizer = PositionSummarizer('last', normalize=False)

    assert list_equal(summarizer.predict(input), [0, 1, 2])


def test_last_normalized():
    summarizer = PositionSummarizer('last')

    assert summarizer.predict(['0', '1', '2']) == approx([0, 1 / 3, 2 / 3])


def test_pos_summarizer_unknown_mode():
    with raises(ValueError):
        _ = PositionSummarizer('unknown')


def test_length_summarizer_unknown_mode():
    with raises(ValueError):
        _ = LengthSummarizer('unknown')


def test_first_summ():
    summarizer = PositionSummarizer()

    with tokenizer_context("icu") as Doc:
        doc = Doc("Ali gel. Ayşe gel. Ahmet git.")

        assert summarizer(doc, k=2) == doc[:2]


@pytest.mark.parametrize("input", [['0', '1']])
def test_random_default(input):
    summarizer = RandomSummarizer(normalize=False)

    assert summarizer.predict(input) == approx(np.array([0.37454012, 0.95071431]))


def test_random_normalized():
    summarizer = RandomSummarizer(normalize=True)

    assert summarizer.predict(['0', '1']) == approx(np.array([0.28261752, 0.71738248]))


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
def test_token_length_summarizer_bert():
    summarizer = LengthSummarizer(normalize=False)

    with tokenizer_context("bert") as Doc:
        assert summarizer.predict(
            list(Doc(
                'Meksikalılaştıramadıklarımızdan olduğunuz kesin. Meksikalı olmadığımızın bilincindeyim.'))) == approx(
            np.array([9, 7]))


def test_token_length_summarizer_icu():
    summarizer = LengthSummarizer(normalize=False)

    with tokenizer_context("icu") as Doc:
        assert summarizer.predict(
            list(Doc(
                'Meksikalılaştıramadıklarımızdan olduğunuz kesin. Meksikalı olmadığımızın bilincindeyim.'))) == approx(
            np.array([4, 4]))


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
def test_token_length_summarizer_normalized():
    summarizer = LengthSummarizer(normalize=True)

    with tokenizer_context("bert") as Doc:
        assert summarizer.predict(
            list(Doc(
                'Meksikalılaştıramadıklarımızdan olduğunuz kesin. Meksikalı olmadığımızın bilincindeyim.'))) == approx(
            np.array([9 / 16, 7 / 16]))


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
def test_char_length_summarizer():
    summarizer = LengthSummarizer(mode="char", normalize=False)

    with tokenizer_context("bert") as Doc:
        assert summarizer.predict(
            list(Doc(
                'Meksikalılaştıramadıklarımızdan olduğunuz kesin. Meksikalı olmadığımızın bilincindeyim.'))) == approx(
            np.array([56, 42]))


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
def test_token_char_summarizer_normalized():
    summarizer = LengthSummarizer(mode="char", normalize=True)

    with tokenizer_context("bert") as Doc:
        assert summarizer.predict(
            list(Doc(
                'Meksikalılaştıramadıklarımızdan olduğunuz kesin. Meksikalı olmadığımızın bilincindeyim.'))) == approx(
            np.array([56 / 98, 42 / 98]))
