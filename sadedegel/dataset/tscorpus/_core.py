import gzip
import json
import sys
from os.path import expanduser, getsize
from pathlib import Path
from typing import Iterator

from rich.console import Console

console = Console()

tarballs = ["art_culture.jsonl.gz", "education.jsonl.gz", "horoscope.jsonl.gz", "life_food.jsonl.gz",
            "politics.jsonl.gz", "technology.jsonl.gz", "economics.jsonl.gz", "health.jsonl.gz",
            "life.jsonl.gz", "magazine.jsonl.gz", "sports.jsonl.gz", "travel.jsonl.gz"]

CATEGORIES = [tb.replace('.jsonl.gz', '') for tb in tarballs]

CORPUS_SIZE = 302936

__general_download_message__ = """Ensure that you have properly downloaded extended or tokenization corpus using

            python -m sadedegel.dataset.extended download --access-key xxx --secret-key xxxx
            python -m sadedegel.dataset.tokenization download --access-key xxx --secret-key xxxx

        Unfortunately due to data licensing issues we could not share data publicly. 
        Get in touch with sadedegel team to obtain a download key.

        """

__tokenization_download_message__ = """Ensure that you have properly downloaded tokenization corpus using

            python -m sadedegel.dataset.tokenization download --access-key xxx --secret-key xxxx

        Unfortunately due to data licensing issues we could not share data publicly. 
        Get in touch with sadedegel team to obtain a download key.

        """

__tokenization_version_message__ = "Ensure your dataset is in versioned format."

__tscorpus_validate__ = """It seems that your copy of TSCorpus is [red]corrupted[/red]. For validation run
 
            python -m sadedegel.dataset.tscorpus validate
            
        To download corpus again run
        
            python -m sadedegel.dataset.tscorpus download --access-key xxx --secret-key xxxx
"""


def fail_with_corpus_validation():
    console.log(__tscorpus_validate__)
    sys.exit(1)


def check_directory_structure(path: str) -> bool:
    base_dir = Path(expanduser(path))
    if not base_dir.exists():
        console.log(f"Dataset base directory ([bold red]{base_dir}[/bold red]) does not exist")

    elif not (base_dir / "tscorpus" / "raw").exists() or not (base_dir / "tscorpus" / "tokenized").exists():
        console.log(f"TsCorpus directory ([bold red]{base_dir}/tscorpus[/bold red]) does not exist")

    else:
        return True

    console.log(__general_download_message__)

    return False


def load_tscorpus_raw(data_home="~/.sadedegel_data") -> Iterator[str]:
    if not check_directory_structure(data_home):
        raise Exception("Ts Corpus validation error")

    for tarball in tarballs:
        with gzip.open(Path(expanduser(data_home)) / 'tscorpus' / 'raw' / tarball, 'rt') as fp:
            for line in fp:
                yield json.loads(line)


def load_tokenization_raw(data_home="~/.sadedegel_data") -> Iterator[str]:
    for d in load_tscorpus_raw(data_home):
        yield dict(id=d['id'], text=d['text'])


def load_classification_raw(data_home="~/.sadedegel_data") -> Iterator[str]:
    for d in load_tscorpus_raw(data_home):
        yield dict(id=d['id'], category=CATEGORIES.index(d['category']), text=d['text'])


def load_tokenization_tokenized(data_home="~/.sadedegel_data") -> Iterator[str]:
    if check_directory_structure(data_home):
        for tarball in tarballs:
            with gzip.open(Path(expanduser(data_home)) / 'tscorpus' / 'tokenized' / tarball, 'rt') as fp:
                for line in fp:
                    d = json.loads(line)
                    yield dict(id=d['id'], tokens=d['text'])
    else:
        return None


def raw_stats(data_home: str) -> int:
    sz = 0
    for f in (Path(expanduser(data_home)) / "tscorpus" / "raw").glob("*.gz"):
        sz += getsize(f)
    return sz


def tokenized_stats(data_home: str) -> int:
    sz = 0
    for f in (Path(expanduser(data_home)) / "tscorpus" / "tokenized").glob("*.gz"):
        sz += getsize(f)
    return sz


def check_and_display(data_home="~/.sadedegel_data"):
    if check_directory_structure(data_home):
        return dict(byte=dict(raw=raw_stats(data_home),
                              tokens=tokenized_stats(data_home)))
