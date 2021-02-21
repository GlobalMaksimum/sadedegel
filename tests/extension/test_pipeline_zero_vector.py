from pathlib import Path  # pylint: disable=unused-import
from os.path import expanduser  # pylint: disable=unused-import
from itertools import tee, product, islice
from random import randint

import pytest
from sklearn.linear_model import SGDClassifier

from rich.progress import track

from .context import OnlinePipeline, TfidfVectorizer, load_extended_raw_corpus

BATCH_SIZE = 10


@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data")).exists()')
@pytest.mark.parametrize('tf_type, text', product(['freq', 'double_norm'], ['ıyi', '*******', '😗😗😗😗😗😗😗😗😗😗']))
def test_tfidf_vectorizer_smoothing(tf_type, text):
    pipeline = OnlinePipeline([('tfidf', TfidfVectorizer(tf_method=tf_type, idf_method='smooth')),
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


@pytest.mark.parametrize('tf_type, text', product(['freq', 'double_norm'], ['ıyi', '*******', '😗😗😗😗😗😗😗😗😗😗']))
def test_zero_vector_edge_cases(tf_type, text):
    vectorizer = TfidfVectorizer(tf_method=tf_type, idf_method='smooth')
    assert vectorizer.fit_transform([text]).sum() == 0
