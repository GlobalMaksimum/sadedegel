from math import ceil
from pathlib import Path
from os.path import dirname

from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

from rich.console import Console

from joblib import dump

import numpy as np

from ..dataset.tscorpus import load_classification_raw, CATEGORIES, CORPUS_SIZE
from ..extension.sklearn import TfidfVectorizer, OnlinePipeline, Text2Doc

from .util import load_model

console = Console()


def empty_model():
    return OnlinePipeline(
        [('text2doc', Text2Doc("icu")),
         ('tfidf', TfidfVectorizer(tf_method='log_norm', idf_method='smooth', drop_punct=True, drop_stopwords=False,
                                   lowercase=True)),
         ('sgd',
          SGDClassifier(penalty="l2", alpha=0.005190632263776186, loss="log", average=False, fit_intercept=True))])


def build(max_rows=-1, batch_size=10000):
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))

    raw = load_classification_raw()
    df = pd.DataFrame.from_records(raw)
    df = shuffle(df, random_state=42)

    if max_rows > 0:
        df = df.sample(max_rows)

    np.random.seed(42)
    msk = np.random.rand(len(df)) < 0.9

    df_train = df[msk]  # random state is a seed value
    df_test = df[~msk]

    BATCH_SIZE = batch_size

    n_split = ceil(len(df_train) / BATCH_SIZE)
    console.log(f"{n_split} batches of {BATCH_SIZE} instances...")

    batches = np.array_split(df_train, n_split)

    pipeline = empty_model()

    for batch in batches:
        pipeline.partial_fit(batch.text, batch.category, classes=[i for i in range(len(CATEGORIES))])

        y_pred = pipeline.predict(df_test.text)
        console.log(f"Accuracy on test: {accuracy_score(df_test.category, y_pred)}")

    console.log("Model build [green]DONE[/green]")

    model_dir = Path(dirname(__file__)) / 'model'

    model_dir.mkdir(parents=True, exist_ok=True)

    pipeline.steps[0][1].Doc = None

    dump(pipeline, (model_dir / 'news_classification.joblib').absolute(), compress=('gzip', 9))


def load(model_name="news_classification"):
    return load_model(model_name)


if __name__ == '__main__':
    build(-1, batch_size=10_000)
