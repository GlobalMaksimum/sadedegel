from .context import load_raw_corpus, load_sentence_corpus, load_annotated_corpus
import pytest

raw_corpus_parameters = [(load_raw_corpus, True, 98),
                         (load_raw_corpus, False, 98)]

sentence_including_corpus_parameters = [(load_sentence_corpus, True, 98),
                                        (load_sentence_corpus, False, 98),
                                        (load_annotated_corpus, True, 96),
                                        (load_annotated_corpus, False, 96)]

corpus_parameters = raw_corpus_parameters + sentence_including_corpus_parameters


@pytest.mark.parametrize("loader, return_iter, expected_count", corpus_parameters)
def test_corpus_size(loader, return_iter, expected_count):
    docs = loader(return_iter=return_iter)

    assert sum(1 for _ in docs) == expected_count

    if not return_iter:
        assert len(docs) == expected_count


@pytest.mark.parametrize("loader, return_iter, expected_count", sentence_including_corpus_parameters)
def test_sentence_including_corpus_integrity(loader, return_iter, expected_count):
    docs = loader(return_iter=return_iter)

    assert all((('sentences' in doc) for doc in docs))

    if loader == load_annotated_corpus:
        assert all((('relevance' in doc) for doc in docs))
