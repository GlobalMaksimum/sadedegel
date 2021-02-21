import click
import os.path
from loguru import logger
from pathlib import Path
import boto3
from smart_open import open
import tarfile
import glob
from rich.progress import track
from sadedegel import Doc
import json

from ..util import safe_read
from ._core import load_extended_metadata


@click.group(help="Extended Dataset commandline")
def cli():
    pass


@cli.command()
@click.option("--access-key", help="Access Key ID to download dataset.", prompt=True,
              default=lambda: os.environ.get('sadedegel_access_key', ''))
@click.option("--secret-key", help="Secret Key ID to download dataset.", prompt=True,
              default=lambda: os.environ.get('sadedegel_secret_key', ''))
@click.option("--data_home", '-d', help="Data home directory", default="~/.sadedegel_data")
def download(access_key, secret_key, data_home):
    """Download extended corpus from cloud with your key."""

    data_home = Path(os.path.expanduser(data_home)) / 'extended'
    logger.info(f"Data directory for extended data {data_home}")

    data_home.mkdir(parents=True, exist_ok=True)

    transport_params = {
        'session': boto3.Session(aws_access_key_id=access_key,
                                 aws_secret_access_key=secret_key),
        'resource_kwargs': {
            'endpoint_url': 'https://storage.googleapis.com',
        }
    }

    __tarballs__ = ['cumhuriyet.tar.gz', 'haberturk.tar.gz', 'hurriyet.tar.gz', 'milliyet.tar.gz',
                    'yetkin-report.tar.gz']

    click.secho("Downloading corpus...")

    for tarball in __tarballs__:
        url = f"s3://sadedegel/dataset/raw/{tarball}"

        with open(url, 'rb', transport_params=transport_params) as fp:
            tf = tarfile.open(fileobj=fp)

            tf.extractall(data_home.absolute())

        url = f"s3://sadedegel/dataset/sents/{tarball}"

        with open(url, 'rb', transport_params=transport_params) as fp:
            tf = tarfile.open(fileobj=fp)

            tf.extractall(data_home.absolute())

        click.secho(f".done: {tarball}", fg="green")


@cli.command()
@click.option("--data_home", '-d', help="Data home directory", default="~/.sadedegel_data")
def sbd(data_home):
    """Generate sentence boundary detected corpus out of raw document corpus."""

    data_home = Path(os.path.expanduser(data_home))

    logger.info(f"Data directory for extended data {data_home}")

    raw_dir = data_home / 'extended' / 'raw'

    for section in raw_dir.iterdir():

        sents_dir = section.parent.parent / 'sents' / str(section.name)

        if section.is_dir():
            sents_dir.mkdir(parents=True, exist_ok=True)

            for raw in track(glob.glob(str((raw_dir / section / '*.txt').absolute())),
                             description=f"{section.name} documents"):
                fn_noext, _ = os.path.splitext(os.path.basename(raw))

                target = (sents_dir / f"{fn_noext}.json").absolute()

                if not os.path.exists(target) or (os.path.exists(target) and os.path.getsize(target) == 0):
                    try:
                        d = Doc(safe_read(raw))

                        with open(target, 'w') as wp:
                            json.dump(
                                dict(sentences=[s.text for s in d], rouge1=[s.rouge1("f1") for s in d]),
                                wp,
                                ensure_ascii=False)
                    except:
                        logger.exception(f"Error in processing document {raw}")

                        raise


@cli.command()
@click.option("--data_home", '-d', help="Data home directory", default="~/.sadedegel_data")
def metadata(data_home):
    """Check extended dataset corpus metadata"""
    from pprint import pformat

    md = load_extended_metadata(data_home)

    if md is not None:
        click.secho(pformat(md), fg="yellow")


if __name__ == '__main__':
    cli()
