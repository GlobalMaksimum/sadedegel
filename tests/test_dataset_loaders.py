from .context import load_raw_corpus, load_sentence_corpus
import pytest

__corpus_length__ = 98

# MOVED to datasets/test_dataset_loaders
@pytest.mark.skip()
def test_load_raw_iter():
    doc_iter = load_raw_corpus(return_iter=True)

    assert sum(1 for _ in doc_iter) == __corpus_length__


@pytest.mark.skip()
def test_load_raw_list():
    doc_list = load_raw_corpus(return_iter=False)

    assert len(doc_list) == __corpus_length__


@pytest.mark.skip()
def test_load_sent_iter():
    doc_iter = load_sentence_corpus(return_iter=True)

    assert sum(1 for _ in doc_iter) == __corpus_length__


@pytest.mark.skip()
def test_load_sent_list():
    doc_list = load_sentence_corpus(return_iter=False)

    assert len(doc_list) == __corpus_length__


@pytest.mark.skip()
def test_proper_dictionary():
    doc_list = load_sentence_corpus(return_iter=False)

    assert all((('sentences' in doc) for doc in doc_list))
