from math import ceil
from pathlib import Path
from os.path import dirname

import numpy as np
import json

from sadedegel.dataset.tscorpus import load_tscorpus_raw
from sadedegel.extension.sklearn import TfidfVectorizer, OnlinePipeline

from sklearn.decomposition import LatentDirichletAllocation

from rich.console import Console

import joblib

console = Console()


def empty_model(batch_size, sample_size):
    return OnlinePipeline([('tfidf', TfidfVectorizer(tf_method="raw",
                                                     idf_method="smooth")),
                           ('lda', LatentDirichletAllocation(n_components=15,
                                                             batch_size=batch_size,
                                                             total_samples=sample_size,
                                                             n_jobs=-1,
                                                             evaluate_every=10,
                                                             perp_tol=1e-1,
                                                             learning_method='online'))])


def build(train_size=200_000):
    raw = load_tscorpus_raw()

    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))

    df = pd.DataFrame().from_records(raw)
    df = df.sample(train_size)

    BATCH_SIZE = 100

    n_split = ceil(len(df) / BATCH_SIZE)
    console.log(f"{n_split} batches of {BATCH_SIZE} instances...")

    batches = np.array_split(df, n_split)

    pipeline = empty_model(BATCH_SIZE, train_size)

    for batch in batches:
        pipeline.partial_fit(batch.text)

    console.log("Model build [green]DONE[/green]")

    model_dir = Path(dirname(__file__)) / 'models'

    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, (model_dir / 'lda_tscorpus_raw_smooth.joblib').absolute(), compress=('gzip', 9))


def load(pipeline_name="lda_tscorpus_raw_smooth.joblib"):
    model_dir = Path(dirname(__file__)) / 'models'
    return joblib.load((model_dir / pipeline_name).absolute())


def display_topic_words():
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))

    lda_model = load()['lda']
    topic_top_token_ix = np.argsort(lda_model.components_, axis=1)[:, -10:]

    vocab_path = Path(dirname(__file__)) / '..' / '..' / 'bblock' / 'data' / 'vocabulary.json'
    with open(vocab_path.absolute(), 'r') as j:
        vocab = json.load(j)

    vocab_df = pd.DataFrame().from_records(vocab['words'])

    for t, topic in enumerate(topic_top_token_ix):
        console.log(f"Topic{t}: {vocab_df.loc[vocab_df.id.isin(topic), 'word'].values}")


def get_topic_vector(ix: int):
    lda_model = load()['lda']
    return lda_model.components_[ix]


if __name__ == "__main__":
    build()
