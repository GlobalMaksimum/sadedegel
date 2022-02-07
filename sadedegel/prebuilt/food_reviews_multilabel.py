from os.path import dirname
from pathlib import Path

from joblib import dump
from rich.console import Console
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier

from ..extension.sklearn import Text2Doc, HashVectorizer
from .util import load_model

from ..dataset.food_review import load_food_review_train, load_food_review_test

console = Console()


def empty_model():
    return Pipeline(
        [('text2doc', Text2Doc('icu')),
         ('hash', HashVectorizer(n_features=550000, alternate_sign=False)),
         ('sgd', MultiOutputClassifier(
             LogisticRegression(penalty='l1', solver='liblinear', C=0.41850078919961053, fit_intercept=True)))
         ]
    )


cols = ['speed', 'service', 'flavour']


def build(max_rows=-1, save=True):
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))

    raw = load_food_review_train()
    df = pd.DataFrame.from_records(raw)
    df = shuffle(df, random_state=42)

    for col in cols:
        df[col] = df[col].apply(lambda x: 0 if x <= 7 else 1)

    if max_rows > 0:
        df = df.sample(max_rows, random_state=42)

    console.log(f"Corpus Size: {len(df)}")

    pipeline = empty_model()

    pipeline.fit(df.text, df[cols])

    console.log("Model build [green]DONE[/green]")

    if save:
        model_dir = Path(dirname(__file__)) / 'model'

        model_dir.mkdir(parents=True, exist_ok=True)

        pipeline.steps[0][1].Doc = None

        dump(pipeline, (model_dir / 'food_sentiment_multilabel.joblib').absolute(), compress=('gzip', 9))


def load(model_name='food_sentiment_multilabel'):
    return load_model(model_name)


def evaluate(scoring='f1', max_rows=-1):
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))
    model = load()

    raw_test = load_food_review_test()
    test = pd.DataFrame.from_records(raw_test)
    for col in cols:
        test[col] = test[col].apply(lambda x: 0 if x <= 7 else 1)
    true_labels = test[cols]

    if max_rows > 0:
        test = test[:max_rows]
        true_labels = true_labels[:max_rows]

    if scoring == 'f1':
        y_pred = model.predict(test.text)
        console.log(f"Model test accuracy (f1-macro): {f1_score(true_labels, y_pred, average='macro')}")
    if scoring == 'auc':
        import numpy as np
        y_probs = model.predict_proba(test.text)
        y_probs = np.transpose([pred[:,1] for pred in y_probs])
        console.log(f"Model test roc_auc score: {roc_auc_score(true_labels, y_probs, average='macro')}")


if __name__ == "__main__":
    build(save=True)
