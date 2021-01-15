from math import ceil

import numpy as np
from joblib import dump, load as jl_load

from pathlib import Path
from os.path import dirname

from rich.console import Console
from rich.progress import Progress

from sklearn.linear_model import SGDClassifier
from sadedegel.extension.sklearn import TfidfVectorizer, OnlinePipeline

from sklearn.utils import compute_class_weight

console = Console()

_classes = {0: 'PROPER',
            1: 'PROFANE'}


def empty_model(cw='balanced'):
    return OnlinePipeline([("tfidf", TfidfVectorizer(tf_method='freq', idf_method='smooth',
                                                     drop_stopwords=False, drop_suffix=False)),
                           ("sgd_model", SGDClassifier(alpha=1e-7,
                                                       loss='log',
                                                       class_weight=cw,
                                                       n_jobs=-1, random_state=42))])


def build(max_rows=15000):
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))

    datadir = Path(dirname(__file__)) / "data" / "profanity" /"profanity_train.csv.gz"
    df = pd.read_csv(datadir)
    class_weights = compute_class_weight('balanced', classes=list(_classes.keys()), y=df.offensive.values)

    CORPUS_SIZE = len(df)

    if max_rows > 0:
        df = df.sample(max_rows)

    CLASS_WEIGHT = {0: class_weights[0],
                    1: class_weights[1]}
    BATCH_SIZE = 100

    console.log(f"Corpus Size: {CORPUS_SIZE}")
    console.log(f"Training Size: {len(df)}")

    n_split = ceil(len(df) / BATCH_SIZE)
    console.log(f"{n_split} batches of {BATCH_SIZE} instances...")

    batches = np.array_split(df, n_split)

    pipeline = empty_model(cw=CLASS_WEIGHT)

    for batch in batches:
        pipeline.partial_fit(batch.text, batch.offensive, classes=[i for i in range(len(_classes))])

    console.log("Model build [green]DONE[/green]")

    model_dir = Path(dirname(__file__)) / 'model'

    model_dir.mkdir(parents=True, exist_ok=True)

    dump(pipeline, (model_dir / 'tweet_profanity_v1.joblib').absolute(), compress=('gzip', 9))


def load(model_version='v1'):
    return jl_load(Path(dirname(__file__)) / 'model' / f"tweet_profanity_{model_version}.joblib")


if __name__ == "__main__":
    build()
