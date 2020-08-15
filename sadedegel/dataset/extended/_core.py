from os.path import expanduser, getsize
from pathlib import Path
import json
from typing import Iterator, Tuple
import glob
import click

__download_message__ = """Ensure that you have properly downloaded extended corpus using
         
            python -m sadedegel.dataset.extended download --access-key xxx --secret-key xxxx
            
        Unfortunately due to data licensing issues we could not share data publicly. 
        Get in touch with sadedegel team to obtain a download key.
        
        """


def check_directory_structure(path: str) -> bool:
    if not Path(expanduser(path)).exists():
        click.secho(f"{path} not found.\n", fg="red")
        click.secho(__download_message__, fg="red")

        return False
    elif not (Path(expanduser(path)) / 'extended' / 'raw').exists():
        click.secho(f"raw directory in {path} not found.\n", fg="red")
        click.secho(__download_message__, fg="red")

        return False
    elif not (Path(expanduser(path)) / 'extended' / 'sents').exists():
        click.secho(f"sents directory in {path} not found.\n", fg="red")
        click.secho(__download_message__, fg="red")

        return False
    else:
        return True


def raw_stats(data_home: str) -> Tuple[int, int]:
    n, sz = 0, 0
    for f in glob.glob(str((Path(expanduser(data_home)) / 'extended' / 'raw' / '*' / '*.txt').absolute())):
        n += 1
        sz += getsize(f)

    return n, sz


def sents_stats(data_home: str) -> Tuple[int, int]:
    n, sz = 0, 0
    for f in glob.glob(str((Path(expanduser(data_home)) / 'extended' / 'sents' / '*' / '*.json').absolute())):
        n += 1
        sz += getsize(f)

    return n, sz


def load_extended_metadata(data_home="~/.sadedegel_data"):
    if check_directory_structure(data_home):

        raw_count, raw_bytes = raw_stats(data_home)
        sents_count, sents_bytes = sents_stats(data_home)

        return dict(count=dict(raw=raw_count, sents=sents_count), byte=dict(raw=raw_bytes, sents=sents_bytes))
    else:
        return None


def load_extended_raw_corpus(data_home="~/.sadedegel_data") -> Iterator[str]:
    if check_directory_structure(data_home):
        for f in glob.glob(str((Path(expanduser(data_home)) / 'extended' / 'raw' / '*' / '*.txt').absolute())):
            with open(f) as fp:
                yield fp.read()
    else:
        return None


def load_extended_sents_corpus(data_home="~/.sadedegel_data") -> Iterator[dict]:
    if check_directory_structure(data_home):
        for f in glob.glob(str((Path(expanduser(data_home)) / 'extended' / 'sents' / '*' / '*.json').absolute())):
            with open(f) as fp:
                yield json.load(fp)
    else:
        return None
