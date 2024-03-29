import os.path
import sys
from itertools import tee
from pathlib import Path
from zipfile import ZipFile

import boto3
import click
from loguru import logger
from rich.console import Console
from smart_open import open

from ._core import load_product_sentiment_train, CORPUS_SIZE

console = Console()

logger.disable("sadedegel")


@click.group(help="Product Sentiment Dataset Commandline")
def cli():
    pass


@cli.command()
@click.option("--access-key", help="Access Key ID to download dataset.", prompt=True,
              default=lambda: os.environ.get('sadedegel_access_key', ''))
@click.option("--secret-key", help="Secret Key ID to download dataset.", prompt=True,
              default=lambda: os.environ.get('sadedegel_secret_key', ''))
@click.option("--data_home", '-d', help="Data home directory", default="~/.sadedegel_data")
def download(access_key, secret_key, data_home):
    """Download product sentiment corpus from cloud with your key."""

    data_home = Path(os.path.expanduser(data_home)) / 'product_sentiment'
    data_home.mkdir(parents=True, exist_ok=True)
    console.print(f"Data directory for product sentiment data {data_home}")

    transport_params = {
        'session': boto3.Session(aws_access_key_id=access_key,
                                 aws_secret_access_key=secret_key),
        'resource_kwargs': {
            'endpoint_url': 'https://storage.googleapis.com',
        }
    }

    url = f"s3://sadedegel/dataset/product_sentiment/product_sentiment.zip"

    with open(url, 'rb', transport_params=transport_params) as fp:
        with ZipFile(fp) as zp:
            zp.extractall(data_home)


@cli.command()
def validate():
    """Sanity check on corpus
    """
    with console.status("[bold yellow]Validating train"):
        train = load_product_sentiment_train()

        train_clone, train = tee(train, 2)

        n_train = sum(1 for _ in train_clone)
        categories = set([row['sentiment_class'] for row in train])

        if n_train == CORPUS_SIZE:
            console.log("Cardinality check [yellow]DONE[/yellow]")
        else:
            console.log(f"Cardinality check [red]FAILED[/red]")
            console.log(f"|Product sentiment (train)| : {n_train} ({CORPUS_SIZE} expected)")
            sys.exit(1)

        if categories == {0, 1, 2}:
            console.log("Label check [yellow]DONE[/yellow]")
        else:
            console.log("Class check [red]FAILED[/red]")
            console.log(
                f"\tProduct sentiment classes : {categories} ({set(['POSITIVE', 'NEGATIVE', 'NEUTRAL'])} expected)")
            sys.exit(1)


if __name__ == "__main__":
    cli()
