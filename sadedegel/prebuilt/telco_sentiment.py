from os.path import dirname
from pathlib import Path

from joblib import dump
from rich.console import Console
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils import shuffle

from .util import load_model
from ..dataset.telco_sentiment import load_telco_sentiment_train, load_telco_sentiment_test, \
    load_telco_sentiment_target, CORPUS_SIZE
from ..extension.sklearn import HashVectorizer, Text2Doc

console = Console()


def empty_model():
    return Pipeline(
        [('text2doc', Text2Doc("icu")),
         ('hash', HashVectorizer(n_features=496485, alternate_sign=True)),
         ('svc', SVC(C=0.3184661147229449, kernel="linear", verbose=True, random_state=42, probability=True))]
    )


def build(save=True):
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))

    raw = load_telco_sentiment_train()
    df = pd.DataFrame.from_records(raw)
    df = shuffle(df)

    console.log(f"Corpus Size: {CORPUS_SIZE}")

    pipeline = empty_model()

    pipeline.fit(df.tweet, df.sentiment_class)

    evaluate(pipeline)

    console.log("Model build [green]DONE[/green]")

    if save:
        model_dir = Path(dirname(__file__)) / 'model'

        model_dir.mkdir(parents=True, exist_ok=True)

        pipeline.steps[0][1].Doc = None

        dump(pipeline, (model_dir / 'telco_sentiment_classification.joblib').absolute(), compress=('gzip', 9))

        console.log("Model save [green]DONE[/green]")


def load(model_name="telco_sentiment_classification"):
    return load_model(model_name)


def evaluate(model=None):
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))

    if model is None:
        model = load()

    test = pd.DataFrame.from_records(load_telco_sentiment_test())
    testLabel = pd.DataFrame.from_records(load_telco_sentiment_target())

    test = test.merge(testLabel, on='id')

    y_pred = model.predict(test.tweet)

    console.log(f"Model test accuracy (f1-macro): {f1_score(test.sentiment_class, y_pred, average='macro')}")


if __name__ == "__main__":
    build(save=True)
