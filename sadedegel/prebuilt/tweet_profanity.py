from math import ceil
from os.path import dirname
from pathlib import Path

import numpy as np
from joblib import dump, load as jl_load
from rich.console import Console
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

from sadedegel.dataset.profanity import load_offenseval_train, load_offenseval_test, \
    load_offenseval_test_label, CORPUS_SIZE, CLASS_VALUES
from sadedegel.extension.sklearn import TfidfVectorizer, OnlinePipeline

console = Console()


def empty_model():
    return OnlinePipeline(
        [('tfidf', TfidfVectorizer(tf_method='freq', idf_method='smooth')),
         ('pa', PassiveAggressiveClassifier(C=8.770192177653856, average=True))])


def build():
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))

    raw = load_offenseval_train()
    df = pd.DataFrame.from_records(raw)
    df = shuffle(df)

    BATCH_SIZE = 1000

    console.log(f"Corpus Size: {CORPUS_SIZE}")

    n_split = ceil(len(df) / BATCH_SIZE)
    console.log(f"{n_split} batches of {BATCH_SIZE} instances...")

    batches = np.array_split(df, n_split)

    pipeline = empty_model()

    for batch in batches:
        pipeline.partial_fit(batch.tweet, batch.profanity_class, classes=[i for i in range(len(CLASS_VALUES))])

    console.log("Model build [green]DONE[/green]")

    model_dir = Path(dirname(__file__)) / 'model'

    model_dir.mkdir(parents=True, exist_ok=True)

    dump(pipeline, (model_dir / 'tweet_profanity_classification.joblib').absolute(), compress=('gzip', 9))


def load():
    return jl_load(Path(dirname(__file__)) / 'model' / 'tweet_profanity_classification.joblib')


def evaluate():
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))

    model = load()

    test = pd.DataFrame.from_records(load_offenseval_test())
    testLabel = pd.DataFrame.from_records(load_offenseval_test_label())

    test = test.merge(testLabel, on='id')

    y_pred = model.predict(test.tweet)

    console.log(f"Model test accuracy (f1-macro): {f1_score(test.profanity_class, y_pred, average='macro')}")


if __name__ == "__main__":
    build()
