from math import ceil

import numpy as np
from joblib import dump, load as jl_load

from pathlib import Path
from os.path import dirname

from rich.console import Console
from rich.progress import Progress

from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.metrics import f1_score

from sadedegel.extension.sklearn import BM25Vectorizer, OnlinePipeline
from sadedegel.dataset.profanity import load_offenseval_train, load_offenseval_test, \
    load_offenseval_test_label, CORPUS_SIZE, CLASS_VALUES

console = Console()


def empty_model():
    return OnlinePipeline(
        [('bm25', BM25Vectorizer(tf_method='log_norm', idf_method='probabilistic',
                                 drop_stopwords=True, drop_suffix=False,
                                 drop_punct=False, lowercase=True,
                                 k1=1.7094943902452382, b=0.8044484402765772,
                                 show_progress=True)),
         ('sgd', SGDClassifier(alpha=0.0011130974542755533,
                               penalty='elasticnet',
                               eta0=0.5818024200892028,
                               learning_rate='optimal',
                               class_weight=None))]
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

    BATCH_SIZE = 1000

    console.log(f"Corpus Size: {CORPUS_SIZE}")

    n_split = ceil(len(df) / BATCH_SIZE)
    console.log(f"{n_split} batches of {BATCH_SIZE} instances...")

    batches = np.array_split(df, n_split)

    pipeline = empty_model()

    #for batch in batches:
    #    pipeline.partial_fit(batch.tweet, batch.profanity_class, classes=[i for i in range(len(CLASS_VALUES))])

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