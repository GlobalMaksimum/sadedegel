from math import ceil
from pathlib import Path
from os.path import dirname

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.utils import shuffle

from rich.console import Console
from rich.progress import Progress

from joblib import dump, load as jl_load

import numpy as np

from ..dataset.tscorpus import load_classification_raw, CATEGORIES, CORPUS_SIZE
from ..extension.sklearn import TfidfVectorizer, OnlinePipeline

console = Console()


def empty_model():
    return OnlinePipeline(
        [('tfidf', TfidfVectorizer(tf_method='freq', idf_method='smooth')),
         ('pa', PassiveAggressiveClassifier(C=8.23704, average=False))])


def build(max_rows=100_000):
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))

    raw = load_classification_raw()
    df = pd.DataFrame.from_records(raw)
    df = shuffle(df)

    if max_rows > 0:
        df = df.sample(max_rows)

    BATCH_SIZE = 1000

    n_split = ceil(len(df) / BATCH_SIZE)
    console.log(f"{n_split} batches of {BATCH_SIZE} instances...")

    batches = np.array_split(df, n_split)

    pipeline = empty_model()

    with Progress() as progress:
        building_task = progress.add_task("[blue]Training a classifier for news categories...",
                                          total=max_rows if max_rows > 0 else CORPUS_SIZE)

        for batch in batches:
            pipeline.partial_fit(batch.text, batch.category, classes=[i for i in range(len(CATEGORIES))])

            progress.update(building_task, advance=len(batch))

    console.log("Model build [green]DONE[/green]")

    model_dir = Path(dirname(__file__)) / 'model'

    model_dir.mkdir(parents=True, exist_ok=True)

    dump(pipeline, (model_dir / 'news_classification.joblib').absolute(), compress=('gzip', 9))


def load(model_name="news_classification"):
    return jl_load(Path(dirname(__file__)) / 'model' / f"{model_name}.joblib")


if __name__ == '__main__':
    build()
