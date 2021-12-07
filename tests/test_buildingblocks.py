import pkgutil  # noqa: F401 # pylint: disable=unused-import

import numpy as np
import pytest
from scipy.sparse import isspmatrix_csr

from .context import Doc, BertTokenizer, SimpleTokenizer, ICUTokenizer, tokenizer_context, tf_context, config_context


@pytest.mark.parametrize("string", ["", " ", "\n", "\t", "\n\t"])
def test_emptystring(string):
    empty = Doc(string)

    assert len(empty) == 1
    assert len(empty[0].tokens) == 0


@pytest.mark.parametrize("tokenizer", [ICUTokenizer.__name__, SimpleTokenizer.__name__])
def test_tokens(tokenizer):
    with tokenizer_context(tokenizer) as Doc2:
        d = Doc2("Ali topu tut. Ömer ılık süt iç.")

        s0 = d[0]

        assert s0.tokens == ['Ali', 'topu', 'tut', '.']

        assert s0.tokens_with_special_symbols == ['[CLS]', 'Ali', 'topu', 'tut', '.', '[SEP]']


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
def test_tokens_bert():
    with tokenizer_context(BertTokenizer.__name__) as Doc2:
        d = Doc2("Ali topu tut. Ömer ılık süt iç.")

        s0 = d[0]

        assert s0.tokens == ['Ali', 'topu', 'tut', '.']

        assert s0.tokens_with_special_symbols == ['[CLS]', 'Ali', 'topu', 'tut', '.', '[SEP]']


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
@pytest.mark.parametrize("tokenizer", [BertTokenizer.__name__, SimpleTokenizer.__name__, ICUTokenizer.__name__])
def test_bert_embedding_generation(tokenizer):
    with tokenizer_context(tokenizer) as Doc2:

        d = Doc2("Ali topu tut. Ömer ılık süt iç.")
        assert d.bert_embeddings.shape == (2, 768)


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
@pytest.mark.parametrize("tokenizer", [BertTokenizer.__name__, SimpleTokenizer.__name__, ICUTokenizer.__name__])
def test_bert_document_embedding_generation(tokenizer):
    with tokenizer_context(tokenizer) as Doc2:
        d = Doc2("Ali topu tut. Ömer ılık süt iç.")
        assert d.bert_document_embedding.shape == (1, 768)


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
@pytest.mark.parametrize("tokenizer", [BertTokenizer.__name__, SimpleTokenizer.__name__, ICUTokenizer.__name__])
def test_bert_document_embedding_generation_long(tokenizer):
    with tokenizer_context(tokenizer) as Doc2:
        d = Doc2("Ali " * 1024)
        assert d.bert_document_embedding.shape == (1, 768)


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
def test_pretrained_embedding_generation():
    d = Doc("Merhaba dünya. Biz dostuz. Barış için geldik. Sizi lazerlerimizle eritmeyeceğiz.")
    doc_embs = d.get_pretrained_embedding(architecture="distilbert", do_sents=False)
    sent_embs = d.get_pretrained_embedding(architecture="distilbert", do_sents=True)

    assert doc_embs.shape[0] == 1
    assert sent_embs.shape[0] == 4


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
def test_pretrained_embedding_generation_fail():
    with pytest.raises(NotImplementedError, match=r".*is not a supported architecture type.*"):
        d = Doc("Merhaba dünya. Biz dostuz. Barış için geldik. Sizi lazerlerimizle eritmeyeceğiz.")
        doc_embs = d.get_pretrained_embedding(architecture="electra", do_sents=False)
    doc_embs = None
    assert doc_embs is None


@pytest.mark.parametrize('tf_type', ['binary', 'raw', 'freq', 'log_norm', 'double_norm'])
def test_tfidf_embedding_generation(tf_type):
    with tf_context(tf_type) as D:
        d = D("Ali topu tut. Ömer ılık süt iç.")
        assert d.tfidf_matrix.shape == (2, d.vocabulary.size_cs)


@pytest.mark.parametrize('tf_type', ['binary', 'raw', 'freq', 'log_norm', 'double_norm'])
def test_tfidf_compare_doc_and_sent(tf_type):
    with tf_context(tf_type):
        d = Doc("Ali topu tut. Ömer ılık süt iç.")

        for i, sent in enumerate(d):
            assert np.all(
                np.isclose(d.tfidf_matrix.toarray()[i, :], sent.tfidf))


testdata = [(True, True),
            (True, False),
            (False, False),
            (False, True)]


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
@pytest.mark.parametrize("return_numpy, return_mask", testdata)
def test_padded_matrix(return_numpy, return_mask):
    import torch  # pylint: disable=unrecognized-inline-option, import-outside-toplevel, import-error
    with tokenizer_context("bert") as D:
        d = D("Ali topu tut. Ömer ılık süt iç.")

        inp = np.array([[2, 3744, 9290, 2535, 18, 3, 0],
                        [2, 6565, 17626, 5244, 2032, 18, 3]])

        mask = np.array([[1, 1, 1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1, 1, 1]])

        res = d.padded_matrix(return_numpy, return_mask)

        if return_numpy:
            if return_mask:
                assert np.all(res[0] == inp)
                assert np.all(res[1] == mask)
            else:
                assert np.all(res == inp)
        else:
            if return_mask:
                assert torch.all(res[0] == torch.tensor(inp))  # noqa # pylint: disable=not-callable
                assert torch.all(res[1] == torch.tensor(mask))  # noqa # pylint: disable=not-callable
            else:
                assert torch.all(res == torch.tensor(inp))  # noqa # pylint: disable=not-callable


@pytest.mark.parametrize("test_for", ["text", "str", "strall"])
def test_span(test_for):
    d = Doc("Ali   topu  tut.")

    spans = d.spans

    if test_for == "text":
        assert [s.text for s in spans] == ['Ali', 'topu', 'tut.']

    elif test_for == "str":
        assert [str(s) for s in spans] == ['Ali', 'topu', 'tut.']
    else:
        assert str(spans) == "[Ali, topu, tut.]"


def test_doc_with_no_sentence():
    raw = "söz konusu adreste bulunan yolda yağmurdan dolayı çamur ve toprak bulunmaktadır"

    d = Doc(raw)

    assert d[0].tokens == Doc.from_sentences([("söz konusu adreste bulunan yolda yağmurdan "
                                               "dolayı çamur ve toprak bulunmaktadır")])[0].tokens


def test_doc_index():
    d = Doc("Ali topu tut. Ömer ılık süt iç.")

    assert d[0] == "Ali topu tut."


def test_doc_iter_next():
    d = Doc("Ali topu tut. Ömer ılık süt iç.")

    assert next(iter(d)) == "Ali topu tut."


def test_doc_iter_eq():
    d = Doc("Ali topu tut. Ömer ılık süt iç.")

    for i, sentence in enumerate(d):
        assert d._sents[i] == sentence == d[i]


@pytest.mark.parametrize("lowercase", [True, False])
def test_doc_level_tfidf(lowercase):
    with config_context(lowercase=lowercase) as D:
        d = D("Ali topu tut. Ömer ılık süt iç.")

        if lowercase:
            assert d.tfidf.shape == (d.vocabulary.size,)
        else:
            assert d.tfidf.shape == (d.vocabulary.size_cs,)


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
@pytest.mark.parametrize("method, tf, tfidf", [("binary", 8, 33.06598), ("raw", 9, 34.06601)])
def test_doc_level_tf_idf_value_bert(method, tf, tfidf):
    with config_context(tokenizer="bert", tf__method=method, idf__method="smooth") as Doc_c:
        d = Doc_c("Ali topu tut. Ömer ılık süt iç.")

        assert np.sum(d.tf) == pytest.approx(tf)

        assert np.sum(d.tfidf) == pytest.approx(tfidf)


@pytest.mark.parametrize("method, tf, tfidf", [("binary", 8, 35.65801), ("raw", 9, 36.65804)])
def test_doc_level_tf_idf_value_icu(method, tf, tfidf):
    with config_context(tokenizer="icu", tf__method=method, idf__method="smooth") as Doc_c:
        d = Doc_c("Ali topu tut. Ömer ılık süt iç.")

        assert np.sum(d.tf) == pytest.approx(tf)

        assert np.sum(d.tfidf) == pytest.approx(tfidf)


def test_doc_level_tf_idf_type():
    d = Doc("Ali topu tut. Ömer ılık süt iç.")
    assert isspmatrix_csr(d.tfidf_matrix)
