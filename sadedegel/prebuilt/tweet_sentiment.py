
from math import ceil
from pathlib import Path
from os.path import dirname

from sklearn.linear_model import PassiveAggressiveClassifier

from rich.console import Console
from rich.progress import Progress

from joblib import dump, load as jl_load

import numpy as np

from sadedegel.tweet_sentiment import load_tweet_sentiment_train

from ..extension.sklearn import TfidfVectorizer, OnlinePipeline

console = Console()

def empty_model():
    return OnlinePipeline(
        [('tfidf', TfidfVectorizer(tf_method='freq', idf_method='smooth', show_progress=True)),
        ('pa', PassiveAggressiveClassifier(C=6.438004890835467e-10, average=True))
        ]
    )

def build(max_rows=10000):
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))

    raw = load_tweet_sentiment_train()
    df = pd.DataFrame.from_records(raw)

    if df.isna().sum().sum() > 0:
        console.log((f"[red]Warning![/red] The data has {df.isna().sum().sum()} NaN values."
                     f"sadedegel going to try drop them automatically"))

        try:
            df.dropna(inplace=True)
        except RuntimeWarning:
            console.log(("[red]Tried to drop NaN values but failed[/red]"
                        "Model might not work properly"))


    if max_rows > 0:
        df = df.sample(max_rows, random_state = 0)

    BATCH_SIZE = 1000

    n_split = ceil(len(df) / BATCH_SIZE)
    console.log(f"{n_split} batches of {BATCH_SIZE} instances...")

    batches = np.array_split(df, n_split)

    pipeline = empty_model()

    with Progress() as progress:
        building_task = progress.add_task("[blue]Training a classifier for twitter sentiments...[/blue]",
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

    dump(pipeline, (model_dir / 'tweet_sentiment.joblib').absolute(), compress = ('gzip', 9))
    if errors == []:
        console.log("Model build [green]Without Errors[/green]")
    else:
        console.log("Model build [red]With Errors![/red]")
        print(errors)

def load(model_name = "tweet_sentiment"):
    return jl_load(Path(dirname(__file__)) / 'model' / f"{model_name}.joblib")

if __name__ == '__main__':
    build()