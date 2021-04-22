from itertools import islice, product

import pytest
from sklearn.pipeline import Pipeline

from .context import HashVectorizer, Text2Doc, load_tweet_sentiment_train

p = product(["icu", "simple", "bert"], [(1, 3), (2, 5), (3, 8), (3, 6)], [True, False])


@pytest.mark.parametrize("tokenizer, prefix_range, alternate_sign", p)
def test_hashvect(tokenizer, prefix_range, alternate_sign):
    feng_pipeline = Pipeline([("Text2Doc", Text2Doc(tokenizer=tokenizer)),
                              ("HashVec", HashVectorizer(prefix_range=prefix_range,
                                                         alternate_sign=alternate_sign))])
    X = [rec['tweet'] for rec in islice(load_tweet_sentiment_train(), 10)]
    assert feng_pipeline.fit_transform(X).shape[0] == len(X)


def test_prefix_range_type_error():
    with pytest.raises(ValueError, match=r"prefix_range should be of tuple type.*"):
        _ = Pipeline([("Text2Doc", Text2Doc(tokenizer="icu")),
                      ("HashVec", HashVectorizer(prefix_range=[1, 3]))])
