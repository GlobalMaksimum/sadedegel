from .context import load_raw_corpus


def test_load_raw_iter():
    doc_iter = load_raw_corpus(return_iter=True)

    assert sum(1 for _ in doc_iter) == 100


def test_load_raw_list():
    doc_list = load_raw_corpus(return_iter=False)

    assert len(doc_list) == 100
