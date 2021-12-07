from itertools import product

import pytest
from sklearn.pipeline import Pipeline

from .context import HashVectorizer, Text2Doc

p = product(["icu", "simple", "bert"], [(1, 3), (2, 5), (3, 8), (3, 6)], [True, False])


@pytest.mark.parametrize("tokenizer, prefix_range, alternate_sign", p)
def test_hashvect(tokenizer, prefix_range, alternate_sign):
    feng_pipeline = Pipeline([("Text2Doc", Text2Doc(tokenizer=tokenizer)),
                              ("HashVec", HashVectorizer(prefix_range=prefix_range,
                                                         alternate_sign=alternate_sign))])
    mini_corpus = ['Sabah bir tweet', 'Öğlen bir başka tweet', 'Akşam bir tweet', '...ve gece son bir tweet']
    assert feng_pipeline.fit_transform(mini_corpus).shape[0] == len(mini_corpus)


def test_prefix_range_type_error():
    with pytest.raises(ValueError, match=r"prefix_range should be of tuple type.*"):
        _ = Pipeline([("Text2Doc", Text2Doc(tokenizer="icu")),
                      ("HashVec", HashVectorizer(prefix_range=[1, 3]))])
