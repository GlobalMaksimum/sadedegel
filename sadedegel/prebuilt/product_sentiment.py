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

from ..dataset.product_sentiment import load_product_sentiment_train, CORPUS_SIZE, CLASS_VALUES
from ..extension.sklearn import TfidfVectorizer, OnlinePipeline, Text2Doc

from itertools import islice

console = Console()


def empty_model():
    return OnlinePipeline(
        [('text2doc', Text2Doc(tokenizer = 'icu')),
         ('tfidf', TfidfVectorizer(tf_method='binary', idf_method='smooth', show_progress=True)),
         ('pa', SGDClassifier(alpha= 0.051599170662053995, penalty= 'l2', eta0=0.8978760982905452,
                              learning_rate= 'optimal'))
         ]
    )

def cv(k=3, max_instances=-1):
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))

    if max_instances > 0:
        raw = islice(load_product_sentiment_train(), max_instances)
    else:
        raw = load_product_sentiment_train()

    df = pd.DataFrame.from_records(raw)
    df = shuffle(df)

    # BATCH_SIZE = CORPUS_SIZE

    kf = KFold(n_splits=k)
    console.log(f"Corpus Size: {CORPUS_SIZE}")

    scores = []

    for train_indx, test_index in kf.split(df):
        train = df.iloc[train_indx]
        test = df.iloc[test_index]

        pipeline = empty_model()
        pipeline.fit(train.text, train.sentiment_class)

        y_pred = pipeline.predict(test.text)

        scores.append(f1_score(test.sentiment_class, y_pred, average='macro'))

        console.log(scores)


def build(max_instances=-1, save=True):
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))

    if max_instances > 0:
        raw = islice(load_product_sentiment_train(), max_instances)
    else:
        raw = load_product_sentiment_train()

    df = pd.DataFrame.from_records(raw)
    df = shuffle(df)

    pipeline = empty_model()
    pipeline.fit(df.text, df.sentiment_class)

    console.log("Model build [green]DONE[/green]")

    if save:
        model_dir = Path(dirname(__file__)) / 'model'

        model_dir.mkdir(parents=True, exist_ok=True)

        pipeline.steps[0][1].Doc = None

        dump(pipeline, (model_dir / 'product_sentiment.joblib').absolute(), compress=('gzip', 9))


def load(model_name="product_sentiment"):
    return jl_load(Path(dirname(__file__)) / 'model' / f"{model_name}.joblib")


if __name__ == '__main__':
    build()