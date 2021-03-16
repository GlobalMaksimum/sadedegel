from pytest import warns
import pytest
from .context import KMeansSummarizer, AutoKMeansSummarizer, DecomposedKMeansSummarizer, Doc, SimpleTokenizer, \
    BertTokenizer, tokenizer_context


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("tokenizer", [SimpleTokenizer.__name__, BertTokenizer.__name__])
@pytest.mark.parametrize("method", [KMeansSummarizer, AutoKMeansSummarizer, DecomposedKMeansSummarizer])
def test_kmeans(normalized, tokenizer, method):
    with tokenizer_context(tokenizer) as Doc:
        d = Doc('ali topu tut. oya ip atla. ahmet topu at.')

        if tokenizer == SimpleTokenizer.__name__:
            with warns(UserWarning, match="Changing tokenizer to"):
                assert len(method(normalize=normalized).predict(d)) == 3
        else:
            assert len(method(normalize=normalized).predict(d)) == 3
