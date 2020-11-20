import click
import os.path
from loguru import logger
from pathlib import Path
import boto3
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
from smart_open import open
import tarfile
from tabulate import tabulate
from loguru import logger

from ._core import check_and_display, tok_eval
from .tokenized import load_corpus as load_tokenized_corpus
from .raw import load_corpus as load_raw_corpus


logger.disable("sadedegel")


@click.group(help="Tokenization Dataset Commandline.")
def cli():
    pass


@cli.command()
@click.option("--access-key", help="Access Key ID to download dataset.", prompt=True,
              default=lambda: os.environ.get('sadedegel_access_key', ''))
@click.option("--secret-key", help="Secret Key ID to download dataset.", prompt=True,
              default=lambda: os.environ.get('sadedegel_secret_key', ''))
@click.option("--data_home", '-d', help="Data home directory", default="~/.sadedegel_data")
@click.option("--data-version", "-v", help="Dataset version", default="v2")
def download(access_key, secret_key, data_home, data_version):
    """Download tokenization corpus from cloud with your key."""

    data_home = Path(os.path.expanduser(data_home)) / 'tokenization'
    logger.info(f"Data directory for tokenization data {data_home}")

    data_home.mkdir(parents=True, exist_ok=True)

    transport_params = {
        'session': boto3.Session(aws_access_key_id=access_key,
                                 aws_secret_access_key=secret_key),
        'resource_kwargs': {
            'endpoint_url': 'https://storage.googleapis.com',
        }
    }

    __tarballs__ = ['raw/80M_raw.tar.gz', 'tokenized/80M_tokenized.tar.gz']

    click.echo(click.style(f"\nStarting Download...", fg="yellow"))

    for tarball in __tarballs__:
        url = f"s3://sadedegel/dataset/tokenization/{data_version}/{tarball}"

        with open(url, 'rb', transport_params=transport_params) as fp:
            tf = tarfile.open(fileobj=fp)

            extract_to = (Path(os.path.expanduser(data_home)) / data_version / tarball.split('/')[0])

            tf.extractall(extract_to.absolute())

        click.secho(f".done: {tarball}", fg="green")

    d = check_and_display('~/.sadedegel_data')
    click.secho("Checking download...")
    click.secho("Dataset stats: " + click.style(f"{d}", fg="yellow"), color='white')


@cli.command()
@click.option("--data_home", '-d', help="Data home directory", default="~/.sadedegel_data")
@click.option('--data-version', "-v", help="Dataset version", default="v2")
def prepare(data_home, data_version):

    __txts__ = ["Raw_Text_80M_Portion_of_TimeLine.txt",
                "Tokenized_Text_80M_Portion_of_TimeLine.txt"]

    click.secho("Reading "+click.style("Raw .txt file.", fg='blue'))
    with open(str(Path(os.path.expanduser(data_home)) / 'tokenization' / data_version / "raw" / __txts__[0]), 'r') as f:
        raw = ''.join(f.readlines())
    f.close()

    click.secho("Parsing "+click.style("Raw data.", fg='blue'))
    soup = BeautifulSoup(raw, 'html.parser')
    documents = []
    for link in tqdm(soup.find_all('text')):
        d = {
            'index': int(link.get('document').split('_')[-1]) - 1000,
            'doc_id': link.get('document'),
            'category': link.get('category'),
            'text': link.get_text()[1:-1]
        }
        documents.append(d)

    click.secho("Saving " + click.style("Raw .json file.", fg='blue'))
    with open(str(Path(os.path.expanduser(data_home)) / 'tokenization' / data_version / "raw" / "Raw.json"), 'w') as j:
        json.dump(documents, j)

    click.secho("Reading " + click.style("Tokenized .txt file.", fg='blue'))
    with open(str(Path(os.path.expanduser(data_home)) / 'tokenization' / data_version / "tokenized" / __txts__[1]),
              'r') as f:
        tok = ''.join(f.readlines())
    f.close()

    click.secho("Parsing " + click.style("Tokenized data.", fg='blue'))
    soup2 = BeautifulSoup(tok, 'html.parser')
    tokens = []
    for link in tqdm(soup2.find_all('text')):
        d = {
            'index': int(link.get('document').split('_')[-1]) - 1000,
            'doc_id': link.get('document'),
            'category': link.get('category'),
            'tokens': link.get_text().split('\n')[1:-1]
        }
        tokens.append(d)

    click.secho("Saving " + click.style("Tokenized .json file.", fg='blue'))
    with open(str(Path(os.path.expanduser(data_home)) / 'tokenization' / data_version / "tokenized" / "Tokenized.json"),
              'w') as j:
        json.dump(tokens, j)


@click.option("--eval-size", "-s", help="Evaluation set size", default=1000)
@click.option("--tokenizers", "-t", help="List of tokenizers to evaluate", default=["bert", "simple"])
@cli.command()
def evaluate(eval_size, tokenizers):
    click.secho(click.style("Loading tokenization corpus...", fg='magenta'))
    raw_docs = load_raw_corpus()
    tokenized_docs = load_tokenized_corpus()
    click.secho(click.style("Creating Eval Set...", fg='magenta'))
    eval_raw, eval_tokenized = [], []
    for raw_doc, tokenized_doc in zip(raw_docs, tokenized_docs):
        ix = raw_doc['index']
        if ix == eval_size:
            break
        eval_raw.append(raw_doc)
        eval_tokenized.append(tokenized_doc)
    click.secho(click.style("Evaluating", fg='magenta'))
    table = tok_eval(eval_raw, eval_tokenized, tokenizers)

    print(
        tabulate(table,
                 headers=['Tokenizer', 'IoU Score'],
                 tablefmt='github',
                 floatfmt=".4f")
    )


if __name__ == "__main__":
    cli()
