import sys
from pathlib import Path

sys.path.insert(0, (Path(__file__) / '..' / '..').absolute())

from sadedegel.dataset import load_raw_corpus  # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.dataset.extended import load_extended_raw_corpus  # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.extension.sklearn import  Text2Doc, TfidfVectorizer, OnlinePipeline, BM25Vectorizer, CharHashVectorizer # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.config import tokenizer_context # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.dataset.tweet_sentiment import load_tweet_sentiment_train # noqa # pylint: disable=unused-import, wrong-import-position
