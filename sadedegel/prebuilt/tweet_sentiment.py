from math import ceil
from os.path import dirname
from pathlib import Path

import numpy as np
from joblib import dump, load as jl_load
from rich.console import Console
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

from ..dataset.tweet_sentiment import load_tweet_sentiment_train, CORPUS_SIZE, CLASS_VALUES
from ..extension.sklearn import Text2Doc, HashVectorizer

from .util import load_model

from sklearn.pipeline import Pipeline

from itertools import islice

console = Console()


def empty_model():
    return Pipeline(
        [('text2doc', Text2Doc("icu")),
         ('hash', HashVectorizer(n_features=932380, alternate_sign=True)),
         ('sgd', SGDClassifier(alpha=0.00041138474018800035, penalty="l2", loss="log", random_state=42))
         ]
    )


def cv(k=3, max_instances=-1):
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))

    if max_instances > 0:
        raw = islice(load_tweet_sentiment_train(), max_instances)
    else:
        raw = load_tweet_sentiment_train()

    df = pd.DataFrame.from_records(raw)
    df = shuffle(df)

    BATCH_SIZE = 1000

    kf = KFold(n_splits=k)
    console.log(f"Corpus Size: {CORPUS_SIZE}")

    scores = []

    for train_indx, test_index in kf.split(df):
        train = df.iloc[train_indx]
        test = df.iloc[test_index]

        n_split = ceil(len(train) / BATCH_SIZE)
        console.log(f"{n_split} batches of {BATCH_SIZE} instances...")

        batches = np.array_split(train, n_split)

        pipeline = empty_model()

        pipeline.fit(train.tweet, train.sentiment_class)

        y_pred = pipeline.predict(test.tweet)

        scores.append(f1_score(test.sentiment_class, y_pred, average="macro"))

        console.log(scores)


def build(max_instances=-1, save=True):
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))

    if max_instances > 0:
        raw = islice(load_tweet_sentiment_train(), max_instances)
    else:
        raw = load_tweet_sentiment_train()

    df = pd.DataFrame.from_records(raw)
    df = shuffle(df)

    BATCH_SIZE = 1000

    console.log(f"Corpus Size: {CORPUS_SIZE}")

    n_split = ceil(len(df) / BATCH_SIZE)
    console.log(f"{n_split} batches of {BATCH_SIZE} instances...")

    batches = np.array_split(df, n_split)

    pipeline = empty_model()

    pipeline.fit(df.tweet, df.sentiment_class)

    console.log("Model build [green]DONE[/green]")

    if save:
        model_dir = Path(dirname(__file__)) / 'model'

        model_dir.mkdir(parents=True, exist_ok=True)

        pipeline.steps[0][1].Doc = None

        dump(pipeline, (model_dir / 'tweet_sentiment.joblib').absolute(), compress=('gzip', 9))


def load(model_name="tweet_sentiment"):
    return load_model(model_name)


if __name__ == '__main__':
    build()
