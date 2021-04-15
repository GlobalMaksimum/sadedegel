from os.path import dirname
from pathlib import Path

from joblib import dump
from rich.console import Console
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline

from ..extension.sklearn import TfidfVectorizer, Text2Doc
from .util import load_model
from ..dataset.movie_sentiment import load_movie_sentiment_train, load_movie_sentiment_test, \
    load_movie_sentiment_test_label, CORPUS_SIZE

console = Console()


def empty_model():
    return Pipeline(
        [('text2doc', Text2Doc('icu')),
         ('tfidf', TfidfVectorizer(tf_method='log_norm', idf_method='smooth', drop_punct=True,
                                   lowercase=True, show_progress=True)),
         ('logreg', LogisticRegression(penalty='l2', C=0.1905922481841914, fit_intercept=True, solver='liblinear'))
         ]
    )


def build(max_rows=-1, save=True):
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))

    raw = load_movie_sentiment_train()
    df = pd.DataFrame.from_records(raw)

    df = shuffle(df, random_state=42)

    if max_rows > 0:
        df = df.sample(max_rows, random_state=42)

    console.log(f"Corpus Size: {CORPUS_SIZE}")

    pipeline = empty_model()

    pipeline.fit(df.text, df.sentiment_class)

    console.log("Model build [green]DONE[/green]")

    if save:
        model_dir = Path(dirname(__file__)) / 'model'

        model_dir.mkdir(parents=True, exist_ok=True)

        pipeline.steps[0][1].Doc = None

        dump(pipeline, (model_dir / 'movie_sentiment.joblib').absolute(), compress=('gzip', 9))


def load(model_name="movie_sentiment"):
    return load_model(model_name)

    return pipeline


def evaluate():
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))
    model = load()

    raw_test = load_movie_sentiment_test()
    test = pd.DataFrame.from_records(raw_test)
    true_labels = pd.DataFrame.from_records(load_movie_sentiment_test_label())

    y_pred = model.predict(test.text)

    console.log(f"Model test accuracy (f1-macro): {f1_score(true_labels.sentiment_class, y_pred, average='macro')}")


if __name__ == '__main__':
    build()
