from math import ceil
from os.path import dirname
from pathlib import Path

import numpy as np
from joblib import dump, load as jl_load
from rich.console import Console
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

from ..dataset.tweet_sentiment import load_tweet_sentiment_train, CORPUS_SIZE, CLASS_VALUES
from ..extension.sklearn import TfidfVectorizer, Text2Doc
from sklearn.pipeline import Pipeline

from itertools import islice

console = Console()


def empty_model():
    return Pipeline(
        [('text2doc', Text2Doc("icu")),
         ('tfidf', TfidfVectorizer(tf_method='freq', idf_method='smooth', drop_punct=False, drop_stopwords=False,
                                   lowercase=True, show_progress=True)),
         ('svc', SVC(C=0.3392899481481453, kernel="linear", random_state=42))
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
    return jl_load(Path(dirname(__file__)) / 'model' / f"{model_name}.joblib")


if __name__ == '__main__':
    build()
