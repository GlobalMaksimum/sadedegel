import pkgutil  # noqa: F401 # pylint: disable=unused-import

import pytest
import numpy as np

from .context import PreTrainedVectorizer, load_tweet_sentiment_train
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


mini_corpus = ['Sabah bir tweet', 'Öğlen bir başka tweet', 'Akşam bir tweet', '...ve gece son bir tweet']


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
def test_vectorizer_pipeline():
    vecpipe = Pipeline([("bert_doc_embeddings",
                         PreTrainedVectorizer(model="distilbert", do_sents=False, progress_tracking=False))])
    embs = vecpipe.fit_transform(mini_corpus)

    assert embs.shape[0] == len(mini_corpus)


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
@pytest.mark.parametrize("do_sents", [True, False])
def test_vectorizer_sparse_to_array(do_sents):
    vecpipe = Pipeline([("bert_doc_embeddings",
                         PreTrainedVectorizer(model="distilbert", do_sents=do_sents, progress_tracking=False))])
    embs = vecpipe.fit_transform(mini_corpus)
    embs_npy = embs.toarray()

    assert isinstance(embs_npy, np.ndarray)
    assert embs_npy.shape[0] == embs.shape[0]
    assert embs_npy.shape[1] == embs.shape[1]


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
def test_pipeline_fit():
    vec = PreTrainedVectorizer(model="distilbert", do_sents=False, progress_tracking=False)
    lr = LogisticRegression(C=0.123)

    vecpipe = Pipeline([("distilbert_embeddings", vec),
                        ("logreg", lr)])
    X = ["harika bir ürün. kargo da çabuk ulaştı.", "kahve makinam diyarbakıra gitmiş. yapacağınız işi sikeyim."] * 5
    y = [1.0, 0.0] * 5

    preds = vecpipe.fit(X, y).predict(X)
    assert isinstance(preds[0], float)

    cv_scorer = cross_val_score(vecpipe, X=X, y=y, scoring="f1_macro")
    assert isinstance(cv_scorer, np.ndarray)
