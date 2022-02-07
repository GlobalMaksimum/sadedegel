import pkgutil  # noqa: F401 # pylint: disable=unused-import

from pathlib import Path  # pylint: disable=unused-import
from os.path import expanduser  # pylint: disable=unused-import
from pytest import approx
from itertools import tee, product, islice
from random import randint

import pytest
from sklearn.linear_model import SGDClassifier

from rich.progress import track

from .context import OnlinePipeline, TfidfVectorizer, load_extended_raw_corpus, Text2Doc, tokenizer_context

BATCH_SIZE = 10


@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data")).exists()')
@pytest.mark.parametrize('tf_type, text', product(['freq', 'double_norm'], ['ıyi', '*******', '😗😗😗😗😗😗😗😗😗😗']))
def test_tfidf_vectorizer_smoothing(tf_type, text):
    pipeline = OnlinePipeline([('t2d', Text2Doc()),
                               ('tfidf', TfidfVectorizer(tf_method=tf_type, idf_method='smooth')),
                               ('model', SGDClassifier())])

    X1 = islice(load_extended_raw_corpus(), 101)

    batch = []
    for doc in X1:
        batch.append(doc)

        if len(batch) == BATCH_SIZE:
            pipeline.partial_fit(batch, [randint(0, 1) for _ in range(len(batch))], classes=[0, 1])

            batch.clear()

    pipeline.partial_fit(batch, [1 for _ in range(len(batch))])

    assert pipeline.predict([text]) > -1  # Can perform inference without value error.


@pytest.mark.parametrize('tf_type, text', product(['freq', 'double_norm'], ['ıyi', '*******']))
def test_zero_vector_edge_cases(tf_type, text):
    vectorizer = TfidfVectorizer(tf_method=tf_type, idf_method='smooth')

    with tokenizer_context("icu") as Doc:
        assert vectorizer.fit_transform([Doc(text)]).sum() == 0


@pytest.mark.skip()
@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
@pytest.mark.parametrize('tf_type, text', product(['freq', 'double_norm'], ['😗😗😗😗😗😗😗😗😗😗']))
def test_zero_vector_edge_case_emoji(tf_type, text):
    vectorizer = TfidfVectorizer(tf_method=tf_type, idf_method='smooth')

    with tokenizer_context("bert") as Doc:
        assert vectorizer.fit_transform([Doc(text)]).sum() == approx(4.479194)


@pytest.mark.parametrize('tf_type, text', product(['freq', 'double_norm'], ['😗😗😗😗😗😗😗😗😗😗']))
def test_zero_vector_edge_case_emoji_icu(tf_type, text):
    vectorizer = TfidfVectorizer(tf_method=tf_type, idf_method='smooth')

    with tokenizer_context("icu") as Doc:
        assert vectorizer.fit_transform([Doc(text)]).sum() == 0
