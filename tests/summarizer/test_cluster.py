from pytest import warns, raises
import pytest
from .context import KMeansSummarizer, AutoKMeansSummarizer, DecomposedKMeansSummarizer, Doc, SimpleTokenizer, \
    BertTokenizer, tokenizer_context


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("tokenizer", [SimpleTokenizer.__name__, BertTokenizer.__name__])
@pytest.mark.parametrize("method", [KMeansSummarizer, AutoKMeansSummarizer, DecomposedKMeansSummarizer])
@pytest.mark.parametrize("embedding", ["bert", "word2vec", "xlnet"])
def test_kmeans(normalized, tokenizer, method, embedding):
    with tokenizer_context(tokenizer):
        d = Doc('ali topu tut. oya ip atla. ahmet topu at.')

        if embedding not in ['bert', 'word2vec']:
            with raises(ValueError, match=r".*is not a valid embedding type.*"):
                method(normalize=normalized, embedding_type=embedding).predict(d)
        else:
            if tokenizer == SimpleTokenizer.__name__ and embedding == 'bert':
                with warns(UserWarning, match="Changing tokenizer to"):
                    assert len(method(normalize=normalized, embedding_type=embedding).predict(d)) == 3
            if tokenizer == SimpleTokenizer.__name__ and embedding == 'word2vec':
                with tokenizer_context(tokenizer):
                    assert len(method(normalize=normalized, embedding_type=embedding).predict(d)) == 3
            else:
                assert len(method(normalize=normalized, embedding_type=embedding).predict(d)) == 3
