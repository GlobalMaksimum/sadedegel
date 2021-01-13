from math import ceil

import numpy as np
from joblib import dump, load as jl_load

from pathlib import Path
from os.path import dirname

from rich.console import Console
from rich.progress import Progress

from sklearn.naive_bayes import MultinomialNB
from ..extension.sklearn import TfidfVectorizer, OnlinePipeline

console = Console()

_classes = {0: 'NEUTRAL',
            1: 'NEGATIVE',
            2: 'POSITIVE'}


def empty_model():
    return OnlinePipeline([
        ("tfidf", TfidfVectorizer(tf_method='freq', idf_method='smooth',
                                  drop_stopwords=False, lowercase=False,
                                  drop_suffix=False)),
        ("nb_model", MultinomialNB(alpha=0.1805))
    ])


def build(max_rows=7500):
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))

    data_dir = Path(dirname(__file__)) / 'data' / 'customer_satisfaction' / 'customer_train.csv.gz'
    df = pd.read_csv(data_dir)

    if max_rows > 0:
        df = df.sample(max_rows)

    BATCH_SIZE = 100
    CORPUS_SIZE = len(df)

    console.log(f"Corpus Size: {CORPUS_SIZE}")

    n_split = ceil(len(df) / BATCH_SIZE)
    console.log(f"{n_split} batches of {BATCH_SIZE} instances...")

    batches = np.array_split(df, n_split)

    pipeline = empty_model()

    with Progress() as progress:
        building_task = progress.add_task("[blue]Training a classifier for customer sentiments...",
                                          total=max_rows if max_rows > 0 else CORPUS_SIZE)

        for batch in batches:
            try:
                pipeline.partial_fit(batch.text, batch.sentiment, classes=[i for i in range(len(df.sentiment.unique()))])
            except:
                print(batch.text)
            progress.update(building_task, advance=len(batch))

    console.log("Model build [green]DONE[/green]")

    model_dir = Path(dirname(__file__)) / 'model'

    model_dir.mkdir(parents=True, exist_ok=True)

    dump(pipeline, (model_dir / 'customer_sentiment_v1.joblib').absolute(), compress=('gzip', 9))


def load(model_version='v1'):
    return jl_load(Path(dirname(__file__)) / 'model' / f"customer_sentiment_{model_version}.joblib")


if __name__ == '__main__':
    build()
