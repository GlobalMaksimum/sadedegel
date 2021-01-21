import pytest
from pytest import approx, raises
from .context import BandSummarizer, Doc


def list_equal(a: list, b: list) -> bool:
    return all((_a == _b) for _a, _b in zip(a, b))


@pytest.mark.parametrize("input", [['0', '1', '2']])
def test_simple_band(input):
    summarizer = BandSummarizer(k=2, normalize=False)

    assert list_equal(summarizer.predict(input), [2, 0, 1])


@pytest.mark.parametrize("input", [['0', '1', '2', '3']])
def test_simple_band(input):
    summarizer = BandSummarizer(k=2, normalize=False)

    assert list_equal(summarizer.predict(input), [3, 1, 2, 0])


def test_simple_band_normalized():
    summarizer = BandSummarizer(k=2, normalize=True)

    assert summarizer.predict(['0', '1', '2']) == approx([2 / 3, 0, 1 / 3])


def test_band_summarizer_unknown_mode():
    with raises(ValueError):
        _ = BandSummarizer(mode='unknown')


def test_band_summarizer_not_implemented_yet():
    with raises(NotImplementedError):
        BandSummarizer(mode='backward').predict(['0', '1', '2'])


def test_band():
    summarizer = BandSummarizer()

    doc = Doc("Ali gel. Ayşe gel. Ahmet git.")

    assert summarizer(doc, k=2) == doc[:2]
