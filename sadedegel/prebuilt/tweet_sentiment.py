from math import ceil
from os.path import dirname
from pathlib import Path

import numpy as np
from joblib import dump, load as jl_load
from rich.console import Console
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

from ..dataset.tweet_sentiment import load_tweet_sentiment_train, CORPUS_SIZE, CLASS_VALUES
from ..extension.sklearn import TfidfVectorizer, OnlinePipeline

console = Console()


def empty_model():
    return OnlinePipeline(
        [('tfidf', TfidfVectorizer(tf_method='freq', idf_method='smooth', show_progress=True)),
         ('pa', PassiveAggressiveClassifier(C=6.438004890835467e-10, average=True))
         ]
    )


def cv(k=3):
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))

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

        for batch in batches:
            pipeline.partial_fit(batch.tweet, batch.sentiment_class,
                                 classes=[i for i in range(len(CLASS_VALUES))])

        y_pred = pipeline.predict(test.tweet)

        scores.append(f1_score(test.sentiment_class, y_pred))

        console.log(scores)


def build():
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))

    raw = load_tweet_sentiment_train()
    df = pd.DataFrame.from_records(raw)
    df = shuffle(df)

    BATCH_SIZE = 1000

    console.log(f"Corpus Size: {CORPUS_SIZE}")

    n_split = ceil(len(df) / BATCH_SIZE)
    console.log(f"{n_split} batches of {BATCH_SIZE} instances...")

    batches = np.array_split(df, n_split)

    pipeline = empty_model()

    for batch in batches:
        pipeline.partial_fit(batch.tweet, batch.sentiment_class,
                             classes=[i for i in range(len(CLASS_VALUES))])

    console.log("Model build [green]DONE[/green]")

    model_dir = Path(dirname(__file__)) / 'model'

    model_dir.mkdir(parents=True, exist_ok=True)

    dump(pipeline, (model_dir / 'tweet_sentiment.joblib').absolute(), compress=('gzip', 9))


def load(model_name="tweet_sentiment"):
    return jl_load(Path(dirname(__file__)) / 'model' / f"{model_name}.joblib")


if __name__ == '__main__':
    build()
