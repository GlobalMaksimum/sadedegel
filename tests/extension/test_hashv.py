from sklearn.pipeline import Pipeline
import pytest
from itertools import product
import pandas as pd
from .context import HashVectorizer, Text2Doc, load_tweet_sentiment_train

p = product(["icu", "simple", "bert"], [(1, 3), [2, 5], (3, 8), [3, 6]], [True, False])


@pytest.mark.parametrize("tokenizer, prefix_range, alternate_sign", p)
def test_hashvect(tokenizer, prefix_range, alternate_sign):
    feng_pipeline = Pipeline([("Text2Doc", Text2Doc(tokenizer=tokenizer)),
                              ("HashVec", HashVectorizer(prefix_range=prefix_range,
                                                         alternate_sign=alternate_sign))])

    df = pd.DataFrame().from_records(load_tweet_sentiment_train()).sample(10, random_state=42)

    assert feng_pipeline.fit_transform(df["tweet"]).shape[0] == len(df)
