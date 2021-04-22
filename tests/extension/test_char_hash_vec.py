from itertools import product
import pytest
from sklearn.pipeline import Pipeline
import pandas as pd

from .context import CharHashVectorizer, Text2Doc, load_tweet_sentiment_train

d = product(["icu", "simple", "bert"], [(1, 1), (1, 3), (3, 6), (4, 7)], [True, False])


@pytest.mark.parametrize("tokenizer, ngram_range, alternate_sign", d)
def test_char_hash_vec(tokenizer, ngram_range, alternate_sign):
    feng_pipeline = Pipeline([("text2doc", Text2Doc(tokenizer=tokenizer)),
                               ("hashvec", CharHashVectorizer(ngram_range=ngram_range,
                                                              alternate_sign=alternate_sign))])

    df = pd.DataFrame().from_records(load_tweet_sentiment_train()).sample(50, random_state=42)

    assert df.shape[0] == feng_pipeline.fit_transform(df.tweet).shape[0]
