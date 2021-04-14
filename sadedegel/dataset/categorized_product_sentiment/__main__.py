import gzip
import os.path
import sys
from itertools import tee
from pathlib import Path
from shutil import copyfileobj

import boto3
import click
from loguru import logger
from rich.console import Console
from smart_open import open

from ._core import load_categorized_product_sentiment_train, CORPUS_SIZE

console = Console()

logger.disable("sadedegel")


@click.group(help="Categorized Product Sentiment Dataset Commandline")
def cli():
    pass


@cli.command()
@click.option("--access-key", help="Access Key ID to download dataset.", prompt=True,
              default=lambda: os.environ.get('sadedegel_access_key', ''))
@click.option("--secret-key", help="Secret Key ID to download dataset.", prompt=True,
              default=lambda: os.environ.get('sadedegel_secret_key', ''))
@click.option("--data_home", '-d', help="Data home directory", default="~/.sadedegel_data")
def download(access_key, secret_key, data_home):
    """Download twitter sentiment corpus from cloud with your key."""

    data_home = Path(os.path.expanduser(data_home)) / 'categorized_product_sentiment'
    data_home.mkdir(parents=True, exist_ok=True)
    console.print(f"Data directory for categorized product sentiment corpus {data_home}")

    transport_params = {
        'session': boto3.Session(aws_access_key_id=access_key,
                                 aws_secret_access_key=secret_key),
        'resource_kwargs': {
            'endpoint_url': 'https://storage.googleapis.com',
        }
    }

    url = f"s3://sadedegel/dataset/categorized_product_sentiment.csv.gz"

    with open(url, 'rb', transport_params=transport_params) as fp, gzip.open(data_home / os.path.basename(url),
                                                                             "wb") as wp:
        copyfileobj(fp, wp)


@cli.command()
def validate():
    """Sanity check on corpus
    """
    with console.status("[bold yellow]Validating train"):
        train = load_categorized_product_sentiment_train()

        train_clone, train_s, train_c = tee(train, 3)

        n_train = sum(1 for _ in train_clone)
        sentiment_categories = set([row['sentiment_class'] for row in train_s])
        product_categories = set([row['product_category'] for row in train_c])

        if n_train == CORPUS_SIZE:
            console.log("Cardinality check [yellow]DONE[/yellow]")
        else:
            console.log(f"Cardinality check [red]FAILED[/red]")
            console.log(f"|Categorized sentiment (train)| : {n_train} ({CORPUS_SIZE} expected)")
            sys.exit(1)

        if sentiment_categories == {0, 1}:
            console.log("Sentiment Label check [yellow]DONE[/yellow]")
        else:
            console.log("Sentiment Label check [red]FAILED[/red]")
            console.log(f"\tCatgorized product sentiment classes : {sentiment_categories} ({set(['POSITIVE', 'NEGATIVE'])} expected)")
            sys.exit(1)

        if product_categories == {0, 1, 2, 3}:
            console.log("Product Category Label check [yellow]DONE[/yellow]")
        else:
            console.log("Category Label check [red]FAILED[/red]")
            console.log(f"\tCatgorized product category classes : {product_categories} ({set(['Kitchen', 'DVD', 'Books', 'Electronics'])} expected)")
            sys.exit(1)


if __name__ == "__main__":
    cli()
