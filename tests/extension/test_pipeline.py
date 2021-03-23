from .context import load_extended_raw_corpus, TfidfVectorizer, OnlinePipeline, load_raw_corpus, Text2Doc

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from rich.progress import track
from itertools import tee, islice
from random import randint

import pytest
from pytest import raises

BATCH_SIZE = 10


def test_pipeline_training():
    pipeline = Pipeline([('t2d', Text2Doc()), ('sg_tfidf', TfidfVectorizer()),
                         ('lr', SGDClassifier())])

    X = load_raw_corpus(False)
    pipeline.fit(X, [randint(0, 1) for _ in range(len(X))])


@pytest.mark.skip()
def test_online_pipeline_training():
    pipeline = OnlinePipeline([('t2d', Text2Doc()),
                               ('sg_tfidf', TfidfVectorizer()),
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
    pipeline = OnlinePipeline([('t2d', Text2Doc()),
                               ('sg_tfidf', TfidfVectorizer()),
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
