import os.path
from pathlib import Path
import sys
from itertools import tee
from smart_open import open
import click
import boto3
from loguru import logger
from ._core import load_food_review_train, load_food_review_test, CLASS_VALUES, TRAIN_SIZE, TEST_SIZE
from zipfile import ZipFile
from rich.console import Console

console = Console()

logger.disable("sadedegel")


@click.group(help="Food Review Dataset Commandline")
def cli():
    pass


@cli.command()
@click.option("--access-key", help="Access Key ID to download dataset.", prompt=True,
              default=lambda: os.environ.get('sadedegel_access_key', ''))
@click.option("--secret-key", help="Secret Key ID to download dataset.", prompt=True,
              default=lambda: os.environ.get('sadedegel_secret_key', ''))
@click.option("--data_home", '-d', help="Data home directory", default="~/.sadedegel_data")
def download(access_key, secret_key, data_home):
    """Download tokenization corpus from cloud with your key."""

    data_home = Path(os.path.expanduser(data_home))
    data_home.mkdir(parents=True, exist_ok=True)
    console.print(f"Data directory for  data {data_home}")

    transport_params = {
        'session': boto3.Session(aws_access_key_id=access_key,
                                 aws_secret_access_key=secret_key),
        'resource_kwargs': {
            'endpoint_url': 'https://storage.googleapis.com',
        }
    }

    url = f"s3://sadedegel/dataset/food_review.zip"

    with open(url, 'rb', transport_params=transport_params) as fp:
        with ZipFile(fp) as zp:
            zp.extractall(data_home)


@cli.command()
def validate():
    """Sanity check on corpus
    """
    with console.status("[bold yellow]Validating train"):
        train = load_food_review_train()

        train_clone, train = tee(train, 2)

        n_train = sum(1 for _ in train_clone)

        if n_train == TRAIN_SIZE:
            console.log("Cardinality check [yellow]DONE[/yellow]")
        else:
            console.log("Cardinality check [red]FAILED[/red]")
            console.log(f"| (train)| : {n_train}")
            sys.exit(1)

    with console.status("[bold yellow]Validate test"):
        test = set((d['id'] for d in load_food_review_test()))

        test_clone, test = tee(test, 2)

        n_test = sum(1 for _ in test_clone)

        if n_test == TEST_SIZE:
            console.log("Cardinality check [yellow]DONE[/yellow]")
        else:
            console.log("Cardinality check [red]FAILED[/red]")
            console.log(f"| (train)| : {n_test}")
            sys.exit(1)


if __name__ == "__main__":
    cli()
