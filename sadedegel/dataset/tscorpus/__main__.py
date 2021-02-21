import os.path
from pathlib import Path
import sys
from itertools import tee

from smart_open import open

import click

import boto3

from loguru import logger

from ._core import load_tokenization_tokenized, load_tokenization_raw, CORPUS_SIZE, tarballs

from rich.console import Console

console = Console()

logger.disable("sadedegel")


@click.group(help="Tokenization Dataset Commandline")
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

    data_home = Path(os.path.expanduser(data_home)) / 'tscorpus'
    console.print(f"Data directory for tokenization data {data_home}")

    transport_params = {
        'session': boto3.Session(aws_access_key_id=access_key,
                                 aws_secret_access_key=secret_key),
        'resource_kwargs': {
            'endpoint_url': 'https://storage.googleapis.com',
        }
    }

    for fmt in ['raw', 'tokenized']:
        (data_home / fmt).mkdir(parents=True, exist_ok=True)

        with console.status(f"[bold green]Downloading {fmt}...") as status:
            for tarball in tarballs:
                url = f"s3://sadedegel/dataset/tscorpus/{fmt}/{tarball}"

                with open(url, 'rb', transport_params=transport_params) as fp, open(data_home / fmt / tarball,
                                                                                    "wb") as wp:
                    wp.write(fp.read())

                console.log(f"{fmt}/{tarball} complete.")


@cli.command()
def validate():
    raw = load_tokenization_raw()
    tok = load_tokenization_tokenized()

    with console.status("[bold yellow]Validating tscorpus"):

        raw_clone, raw = tee(raw, 2)
        tok_clone, tok = tee(tok, 2)

        n_doc_raw = sum(1 for _ in raw_clone)
        n_doc_tok = sum(1 for _ in tok_clone)

        if n_doc_raw != n_doc_tok:
            console.log("Cardinality check [red]FAILED[/red]")
            console.log(f"|TsCorpus (raw)|: {n_doc_raw}")
            console.log(f"|TsCorpus (tokenized)|: {n_doc_tok}")

            sys.exit(1)
        else:
            if n_doc_raw == CORPUS_SIZE:
                console.log("Cardinality check [yellow]DONE[/yellow]")
                n_document = CORPUS_SIZE
            else:
                console.log("Cardinality check [red]FAILED[/red]")
                console.log(f"|TsCorpus (raw)| : {n_doc_raw}")
                sys.exit(1)

        count = 0
        for i, (d1, d2) in enumerate(zip(raw, tok)):
            if d1['id'] != d2['id']:
                console.log("Document order check [red]FAILED[/red]")
                console.log(f"{i}th document (0-based index) raw#:{d1['id']}, tokenized#:{d2['id']}")

                sys.exit(1)

            count += 1
        console.log("Document order check [yellow]DONE[/yellow]")

        if count != n_document:
            console.log(f"[red]Corpus cardinality is {count}/{n_document}. 302936 is expected.")
            sys.exit(1)
        else:

            console.log("TsCorpus is [green]VALID[/green]")


if __name__ == "__main__":
    cli()
