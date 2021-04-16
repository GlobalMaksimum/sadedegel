import os.path
import sys
from itertools import tee
from pathlib import Path
from zipfile import ZipFile
import logging

import boto3
import click
from loguru import logger
from rich.console import Console
from smart_open import open

from ._core import load_test_label, load_test, \
    load_train, CORPUS_SIZE

console = Console()

logger.disable("sadedegel")


@click.group(help="Customer Review Classification Dataset Commandline")
def cli():
    pass


@cli.command()
@click.option("--access-key", help="Access Key ID to download dataset.", prompt=True,
              default=lambda: os.environ.get('sadedegel_access_key', ''))
@click.option("--secret-key", help="Secret Key ID to download dataset.", prompt=True,
              default=lambda: os.environ.get('sadedegel_secret_key', ''))
@click.option("--data_home", '-d', help="Data home directory", default="~/.sadedegel_data")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Cli verbosity")
def download(access_key, secret_key, data_home, verbose):
    """Download tokenization corpus from cloud with your key."""

    data_home = Path(os.path.expanduser(data_home))
    data_home.mkdir(parents=True, exist_ok=True)
    console.print(f"Data directory for customer review classification data {data_home}")

    if verbose:
        boto3.set_stream_logger("boto3", logging.DEBUG)
        boto3.set_stream_logger("botocore", logging.DEBUG)

    transport_params = {
        'session': boto3.Session(aws_access_key_id=access_key,
                                 aws_secret_access_key=secret_key),
        'resource_kwargs': {
            'endpoint_url': 'https://storage.googleapis.com',
        }
    }

    url = f"s3://sadedegel/dataset/customer_review_classification.zip"

    with open(url, 'rb', transport_params=transport_params) as fp:
        with ZipFile(fp) as zp:
            zp.extractall(data_home)


@cli.command()
def validate():
    """Sanity check on corpus
    """
    with console.status("[bold yellow]Validating train"):
        train = load_train()

        train_clone, train = tee(train, 2)

        n_train = sum(1 for _ in train_clone)

        if n_train == CORPUS_SIZE:
            console.log("Cardinality check [yellow]DONE[/yellow]")
        else:
            console.log("Cardinality check [red]FAILED[/red]")
            console.log(f"|Telco Sentiment (train)| : {n_train}")
            sys.exit(1)

    with console.status("[bold yellow]Validate test"):
        test = set((d['id'] for d in load_test()))
        test_label = set((d['id'] for d in load_test_label()))

        a_b, ab, b_a = test - test_label, test & test_label, test_label - test

        if len(a_b) == 0 and len(b_a) == 0:
            console.log("Corpus check [green]DONE[/green]")
        else:
            console.log(f"Test file [red]DIVERGE[/red] from label file. {len(a_b)}, {len(b_a)}")


if __name__ == "__main__":
    cli()
