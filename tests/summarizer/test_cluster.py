from pytest import raises
import pytest
from .context import KMeansSummarizer, AutoKMeansSummarizer, DecomposedKMeansSummarizer, Doc, SimpleTokenizer, \
    BertTokenizer, tokenizer_context


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("tokenizer", [SimpleTokenizer.name, BertTokenizer.name])
@pytest.mark.parametrize("method", [KMeansSummarizer, AutoKMeansSummarizer, DecomposedKMeansSummarizer])
def test_kmeans(normalized, tokenizer, method):
    with tokenizer_context(tokenizer):
        d = Doc('ali topu tut. oya ip atla. ahmet topu at.')

        if tokenizer == SimpleTokenizer.name:
            with raises(NotImplementedError):
                assert len(method(normalize=normalized).predict(d)) == 3
        else:
            assert len(method(normalize=normalized).predict(d)) == 3


@pytest.mark.parametrize("normalized, tokenizer", [[True, False], [SimpleTokenizer.name, BertTokenizer.name]])
def test_autokmeans(normalized, tokenizer):
    with tokenizer_context(tokenizer):
        d = Doc('ali topu tut. oya ip atla. ahmet topu at.')
        assert len(AutoKMeansSummarizer(normalize=normalized).predict(d)) == 3


@pytest.mark.parametrize("normalized, tokenizer", [[True, False], [SimpleTokenizer.name, BertTokenizer.name]])
def test_decomposed_kmeans(normalized, tokenizer):
    with tokenizer_context(tokenizer):
        d = Doc('ali topu tut. oya ip atla. ahmet topu at.')
        assert len(DecomposedKMeansSummarizer(normalize=normalized).predict(d)) == 3


@pytest.mark.parametrize("normalized, tokenizer", [[True, False], [SimpleTokenizer.name, BertTokenizer.name]])
def test_kmeans_parameter_error(normalized, tokenizer):
    with tokenizer_context(tokenizer):
        d = Doc('ali topu tut. oya ip atla. ahmet topu at.')
        assert len(KMeansSummarizer(normalize=normalized).predict(d)) == 3
