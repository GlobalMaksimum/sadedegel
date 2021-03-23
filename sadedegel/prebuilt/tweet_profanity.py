from os.path import dirname
from pathlib import Path

from joblib import dump, load as jl_load
from rich.console import Console

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

from ..extension.sklearn import BM25Vectorizer, Text2Doc

from ..dataset.profanity import load_offenseval_train, load_offenseval_test, \
    load_offenseval_test_label, CORPUS_SIZE
from ..extension.sklearn import OnlinePipeline

console = Console()


def empty_model():
    return OnlinePipeline(
        [('text2doc', Text2Doc("icu")),
         ('bm25', BM25Vectorizer(tf_method='binary', idf_method='smooth',
                                 drop_stopwords=False,
                                 drop_punct=True, lowercase=True, k1=0.02978859422550008, b=4.825535232935995e-05,
                                 delta=1.2258845996051257,
                                 show_progress=True)),
         ('sgd', SGDClassifier(alpha=0.012998795143949122,
                               loss="modified_huber", average=False, penalty="l2",
                               random_state=42, fit_intercept=True))]
    )


def build(save=True):
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

    evaluate(pipeline)

    console.log("Model build [green]DONE[/green]")

    if save:
        model_dir = Path(dirname(__file__)) / 'model'

        model_dir.mkdir(parents=True, exist_ok=True)

        pipeline.steps[0][1].Doc = None

        dump(pipeline, (model_dir / 'tweet_profanity_classification.joblib').absolute(), compress=('gzip', 9))

        console.log("Model save [green]DONE[/green]")


def load():
    return jl_load(Path(dirname(__file__)) / 'model' / 'tweet_profanity_classification.joblib')


def evaluate(model=None):
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))

    if model is None:
        model = load()

    test = pd.DataFrame.from_records(load_offenseval_test())
    testLabel = pd.DataFrame.from_records(load_offenseval_test_label())

    test = test.merge(testLabel, on='id')

    y_pred = model.predict(test.tweet)

    console.log(f"Model test accuracy (f1-macro): {f1_score(test.profanity_class, y_pred, average='macro')}")


if __name__ == "__main__":
    build(save=True)
