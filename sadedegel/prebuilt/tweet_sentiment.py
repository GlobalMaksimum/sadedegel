
from math import ceil
from pathlib import Path
from os.path import dirname

from sklearn.linear_model import SGDClassifier

from rich.console import Console
from rich.progress import Progress

from joblib import dump, load as jl_load

import numpy as np

from ..extension.sklearn import TfidfVectorizer, OnlinePipeline

console = Console()

def empty_model():
    return OnlinePipeline(
        [('tfidf', TfidfVectorizer(tf_method='freq', idf_method='smooth')),
        ('pa', SGDClassifier(alpha=0.0009951711542447474, loss='log'))
        ]
    )

def build(max_rows=3000):
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))


    df = pd.read_csv(Path(dirname(__file__)) / 'data' / 'tweet' / 'tweet_sentiment_train.csv.gz', compression='gzip')

    if max_rows > 0:
        df = df.sample(max_rows, random_state = 0)

    BATCH_SIZE = 1000

    n_split = ceil(len(df) / BATCH_SIZE)
    console.log(f"{n_split} batches of {BATCH_SIZE} instances...")

    batches = np.array_split(df, n_split)

    pipeline = empty_model()

    with Progress() as progress:
        building_task = progress.add_task("[blue]Training a classifier for twitter sentiments...",
                                          total=max_rows if max_rows > 0 else len(df))
        errors = []
        for batch in batches:
            try:
                pipeline.partial_fit(batch.text, batch.sentiment,
                                 classes = df.sentiment.unique().tolist())
            except:
                print(batch.text)
                errors.append(batch.text_uuid)

            progress.update(building_task, advance = len(batch))

    console.log("Model build [green]DONE[/green]")

    model_dir = Path(dirname(__file__)) / 'model'

    model_dir.mkdir(parents = True, exist_ok = True)

    dump(pipeline, (model_dir / 'tweet_sentiment_v1.joblib').absolute(), compress = ('gzip', 9))
    if errors == []:
        console.log("Model build [green]Without Errors[/green]")
    else:
        console.log("Model build [red]With Errors![/red]")
        print(errors)

def load(model_name = "tweet_sentiment_v1"):
    return jl_load(Path(dirname(__file__)) / 'model' / f"{model_name}.joblib")

if __name__ == '__main__':
    build()