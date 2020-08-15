from pytest import raises
import pytest
from .context import KMeansSummarizer, AutoKMeansSummarizer, DecomposedKMeansSummarizer, Doc


@pytest.mark.parametrize("normalized", [True, False])
def test_kmeans(normalized):
    d = Doc('ali topu tut. oya ip atla. ahmet topu at.')

    assert len(KMeansSummarizer(normalize=normalized).predict(d)) == 3


@pytest.mark.parametrize("normalized", [True, False])
def test_autokmeans(normalized):
    d = Doc('ali topu tut. oya ip atla. ahmet topu at.')

    assert len(AutoKMeansSummarizer(normalize=normalized).predict(d)) == 3


@pytest.mark.parametrize("normalized", [True, False])
def test_decomposed_kmeans(normalized):
    d = Doc('ali topu tut. oya ip atla. ahmet topu at.')

    assert len(DecomposedKMeansSummarizer(normalize=normalized).predict(d)) == 3


def test_kmeans_parameter_error():
    d = Doc('ali topu tut. oya ip atla. ahmet topu at.')

    assert len(KMeansSummarizer().predict(d.sents)) == 3
