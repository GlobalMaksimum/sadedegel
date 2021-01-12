from .context import load_extended_raw_corpus, TfidfVectorizer, OnlinePipeline, load_raw_corpus

from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from rich.progress import track
from itertools import tee, islice, product
from random import randint
import numpy as np

import pytest
from pytest import raises

BATCH_SIZE = 10


def test_pipeline_training():
    pipeline = Pipeline([('sg_tfidf', TfidfVectorizer()),
                         ('lr', SGDClassifier())])

    X = load_raw_corpus(False)
    pipeline.fit(X, [randint(0, 1) for _ in range(len(X))])


@pytest.mark.skip()
def test_online_pipeline_training():
    pipeline = OnlinePipeline([('sg_tfidf', TfidfVectorizer()),
                               ('lr', SGDClassifier())])

    X1, X2 = tee(islice(load_extended_raw_corpus(), 101), 2)

    total = sum(1 for _ in X1)

    batch = []
    for doc in track(X2, total=total):
        batch.append(doc)

        if len(batch) == BATCH_SIZE:
            pipeline.partial_fit(batch, [randint(0, 1) for _ in range(len(batch))], classes=[0, 1])

            batch.clear()

    pipeline.partial_fit(batch, [1 for _ in range(len(batch))])

@pytest.mark.skip()
def test_online_pipeline_training_divisable_batch():
    pipeline = OnlinePipeline([('sg_tfidf', TfidfVectorizer()),
                               ('lr', SGDClassifier())])

    X1, X2 = tee(islice(load_extended_raw_corpus(), 100), 2)

    total = sum(1 for _ in X1)

    batch = []

    for doc in track(X2, total=total):
        batch.append(doc)

        if len(batch) == BATCH_SIZE:
            pipeline.partial_fit(batch, [randint(0, 1) for _ in range(len(batch))], classes=[0, 1])

            batch.clear()

    with raises(ValueError, match=r"Ensure that X contains at least one valid document.*"):
        pipeline.partial_fit(batch, [1 for _ in range(len(batch))])


tf_types = ['freq', 'double_norm']
texts = ['ıyi', '*******', '😗😗😗😗😗😗😗😗😗😗']


@pytest.mark.parametrize('tf_type, text', product(tf_types, texts))
def test_tfidf_vectorizer_smoothing(tf_type, text):
    pipeline = OnlinePipeline([('tfidf', TfidfVectorizer(tf_method=tf_type, idf_method='smooth')),
                           ('model', SGDClassifier())])

    X1, X2 = tee(islice(load_extended_raw_corpus(), 101), 2)

    total = sum(1 for _ in X1)

    batch = []
    for doc in track(X2, total=total):
        batch.append(doc)

        if len(batch) == BATCH_SIZE:
            pipeline.partial_fit(batch, [randint(0, 1) for _ in range(len(batch))], classes=[0, 1])

            batch.clear()

    pipeline.partial_fit(batch, [1 for _ in range(len(batch))])

    assert pipeline.predict([text]) > -1  # Can perform inference without value error.


@pytest.mark.parametrize('tf_type, text', product(tf_types, texts))
def test_zero_vector_edge_cases(tf_type, text):
    vectorizer = TfidfVectorizer(tf_method=tf_type, idf_method='smooth')
    assert vectorizer.fit_transform([text]).sum() == 0
