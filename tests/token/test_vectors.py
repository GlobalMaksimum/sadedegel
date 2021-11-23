import pkgutil  # noqa: F401 # pylint: disable=unused-import
import pytest

from .context import Doc, Vocabulary, WordVectorNotFound, tokenizer_context

common_phrase = "Merhaba dünya. Biz dostuz. Barış için geldik."


@pytest.mark.parametrize("tokenizer", ["bert", "simple", "icu"])
def test_indices(tokenizer):
    voc = Vocabulary(tokenizer)

    assert voc.id("calcium") == -1
    assert isinstance(voc.id("dünya"), int)

    if not voc.has_vector("dünya"):
        assert voc.id_to_feature[voc.id("dünya")] == "dünya"
    else:
        assert voc.id_to_feature_has_vector[voc.id_has_vector("dünya")] == "dünya"


@pytest.mark.parametrize("tokenizer", ["simple", "icu"])
def test_vector(tokenizer):
    voc = Vocabulary(tokenizer)
    with tokenizer_context(tokenizer) as Doc2:
        d = Doc2(common_phrase)
        if voc.has_vector("merhaba"):
            assert d.tokens[0].vector.shape[0] == 100
        else:
            with pytest.raises(WordVectorNotFound, match=".*Word Vector not found for.*"):
                vector = d.tokens[0].vector.shape[0] == 100
                assert vector is None


@pytest.mark.skipif('pkgutil.find_loader("torch") is None')
def test_vector_bert():
    voc = Vocabulary("bert")
    with tokenizer_context("bert") as Doc2:
        d = Doc2(common_phrase)
        if voc.has_vector("merhaba"):
            assert d.tokens[0].vector.shape[0] == 100
        else:
            with pytest.raises(WordVectorNotFound, match=".*Word Vector not found for.*"):
                vector = d.tokens[0].vector.shape[0] == 100
                assert vector is None
