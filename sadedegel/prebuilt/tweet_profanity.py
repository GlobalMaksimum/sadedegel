from math import ceil

import numpy as np
from joblib import dump, load as jl_load

from pathlib import Path
from os.path import dirname

from rich.console import Console
from rich.progress import Progress

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.utils import shuffle
from sklearn.metrics import f1_score

from sadedegel.extension.sklearn import BM25Vectorizer, OnlinePipeline
from sadedegel.dataset.profanity import load_offenseval_train, load_offenseval_test, \
    load_offenseval_test_label, CORPUS_SIZE, CLASS_VALUES

console = Console()


def empty_model():
    return OnlinePipeline(
        [('tfidf', BM25Vectorizer(tf_method='log_norm', idf_method='smooth',
                                  drop_stopwords=False, drop_suffix=False,
                                  drop_punct=False, lowercase=True,
                                  k1=1.762598627864453, b=0.4063826957568762)),
         ('pa', PassiveAggressiveClassifier(C=4.4122526140074876e-05, average=True,
                                            class_weight='balanced'))]
    )


def build():
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))

    raw = load_offenseval_train()
    df = pd.DataFrame.from_records(raw)
    df = shuffle(df)

    console.log(f"Corpus Size: {CORPUS_SIZE}")

    pipeline = empty_model()

    pipeline.fit(df.tweet, df.profanity_class)
    console.log("Model build [green]DONE[/green]")

    model_dir = Path(dirname(__file__)) / 'model'

    model_dir.mkdir(parents=True, exist_ok=True)

    dump(pipeline, (model_dir / 'tweet_profanity_classification_v2.joblib').absolute(), compress=('gzip', 9))


def load():
    return jl_load(Path(dirname(__file__)) / 'model' / 'tweet_profanity_classification_v2.joblib')


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
