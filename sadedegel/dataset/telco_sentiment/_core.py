import csv
from pathlib import Path
from rich.console import Console
import gzip

CLASS_VALUES = ['notr', 'olumlu', 'olumsuz']
CORPUS_SIZE = 13832

console = Console()

__general_download_message__ = """Ensure that you have properly downloaded telco sentiment corpus using

            python -m sadedegel.dataset.telco_sentiment download --access-key xxx --secret-key xxxx

        Unfortunately due to data licensing issues we could not share data publicly. 
        Get in touch with sadedegel team to obtain a download key.
        """


def check_directory_structure(path: str) -> bool:
    base_dir = Path(path).expanduser()

    telco_sentiment_dir = base_dir / 'telco_sentiment'

    if not base_dir.exists():
        console.log(f"Dataset base directory ([bold red]{base_dir}[/bold red]) does not exist")

    elif not telco_sentiment_dir.exists():
        console.log(
            f"OffensEval directory ([bold red]{telco_sentiment_dir}[/bold red]) does not exist")

    else:
        return True

    console.log(__general_download_message__)

    return False


def load_telco_sentiment_train(data_home="~/.sadedegel_data"):
    if not check_directory_structure(data_home):
        raise Exception("Telco Sentiment Corpus validation error")

    train_csv = Path(data_home).expanduser() / "telco_sentiment"
    train_csv = train_csv / "telco_sentiment_train.csv.gz"

    with gzip.open(train_csv, "rt") as csvfile:
        rd = csv.DictReader(csvfile)

        for rec in rd:
            yield dict(id=rec['text_uuid'], tweet=rec['tweet'], sentiment_class=CLASS_VALUES.index(rec['sentiment']))


def load_telco_sentiment_test(data_home="~/.sadedegel_data"):
    if not check_directory_structure(data_home):
        raise Exception("Telco Sentiment Corpus validation error")

    test_csv = Path(data_home).expanduser() / "telco_sentiment"
    test_csv = test_csv / "telco_sentiment_test.csv.gz"

    with gzip.open(test_csv, "rt") as csvfile:
        rd = csv.DictReader(csvfile)

        for rec in rd:
            yield dict(id=rec['text_uuid'], tweet=rec['tweet'])


def load_telco_sentiment_test_label(data_home="~/.sadedegel_data"):
    if not check_directory_structure(data_home):
        raise Exception("Telco Sentiment Corpus validation error")

    target_csv = Path(data_home).expanduser() / "telco_sentiment"
    target_csv = target_csv / "telco_sentiment_target.csv.gz"

    with gzip.open(target_csv, "rt") as csvfile:
        rd = csv.DictReader(csvfile)

        for rec in rd:
            yield dict(id=rec['text_uuid'], sentiment_class=CLASS_VALUES.index(rec['sentiment']))
