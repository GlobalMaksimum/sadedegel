from math import ceil
from pathlib import Path
from os.path import dirname

from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline


from rich.console import Console

from joblib import dump

import numpy as np

from ..extension.sklearn import TfidfVectorizer, Text2Doc
from .util import load_model

console = Console()

def empty_model():
    return Pipeline(
        [('text2doc', Text2Doc('icu')),
         ('tfidf', TfidfVectorizer(tf_method='log_norm', idf_method='probabilistic', drop_punct=True,
                                   lowercase=True, show_progress=True)),
         ('logreg', LogisticRegression(penalty='l2', C=0.02987625339246147, fit_intercept=False, solver='liblinear'))
         ]
    )

def build(max_rows=-1, save=True):
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))

    df = pd.read_csv(Path(dirname(__file__)) /  'customer_reviews_train.csv')
    df = shuffle(df, random_state=42)

    if max_rows > 0:
        df = df.sample(max_rows, random_state=42)

    pipeline = empty_model()

    pipeline.fit(df.text, df.review_class)

    if save:
        model_dir = Path(dirname(__file__)) / 'model'

        model_dir.mkdir(parents=True, exist_ok=True)

        pipeline.steps[0][1].Doc = None

        dump(pipeline, (model_dir / 'customer_review_classification.joblib').absolute(), compress=('gzip', 9))

    evaluate(pipeline)

    console.log('Model build [green]DONE[/green]')

def load(model_name='customer_review_classification'):
    return load_model(model_name)

def evaluate(model=None, scoring='f1'):
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))
    model  = load()

    test = pd.read_csv(Path(dirname(__file__)) /  'customer_reviews_test.csv')
    if scoring=='f1':
        y_pred = model.predict(test.text)
        console.log(f"Model test accuracy (f1-macro): {f1_score(test.review_class, y_pred, average='macro')}")
    if scoring=='auc':
        y_pred = model.predict_proba(test.text)
        console.log(f"Model test roc_auc score: {roc_auc_score(test.review_class, y_pred, multi_class='ovr')}")


if __name__ == "__main__":
    build(save=True)