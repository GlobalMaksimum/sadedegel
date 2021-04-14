import csv
import gzip
from pathlib import Path
from rich.console import Console

CLASS_VALUES = ["NEGATIVE", "POSITIVE"]
CORPUS_SIZE = 5800

console = Console()

__general_download_message__ = """Ensure that you have properly downloaded hotel sentiment corpus using

            python -m sadedegel.dataset.hotel_sentiment download --access-key xxx --secret-key xxxx

        Unfortunately due to data licensing issues we could not share data publicly. 
        Get in touch with sadedegel team to obtain a download key.
        """


def check_directory_structure(path: str) -> bool:
    base_dir = Path(path).expanduser()

    hotel_sentiment_dir = base_dir / 'hotel_sentiment'

    if not base_dir.exists():
        console.log(f"Dataset base directory ([bold red]{base_dir}[/bold red]) does not exist")

    elif not hotel_sentiment_dir.exists():
        console.log(
            f" directory ([bold red]{hotel_sentiment_dir}[/bold red]) does not exist")

    else:
        return True

    console.log(__general_download_message__)

    return False


def load_hotel_sentiment_train(data_home="~/.sadedegel_data"):
    if not check_directory_structure(data_home):
        raise Exception(" validation error")

    train_csv = Path(data_home).expanduser() / "hotel_sentiment"
    train_csv = train_csv / "hotel_sentiment_train.csv.gz"

    with gzip.open(train_csv, "rt") as csvfile:
        rd = csv.DictReader(csvfile)

        for rec in rd:
            yield dict(id=rec['text_uuid'], text=rec['text'], sentiment_class=int(rec['sentiment']))


def load_hotel_sentiment_test(data_home="~/.sadedegel_data"):
    if not check_directory_structure(data_home):
        raise Exception("Hotel Sentiment Corpus validation error")

    test_csv = Path(data_home).expanduser() / "hotel_sentiment"
    test_csv = test_csv / "hotel_sentiment_test.csv.gz"

    with gzip.open(test_csv, "rt") as csvfile:
        rd = csv.DictReader(csvfile)

        for rec in rd:
            yield dict(id=rec['text_uuid'], text=rec['text'])


def load_hotel_sentiment_test_label(data_home="~/.sadedegel_data"):
    if not check_directory_structure(data_home):
        raise Exception("Hotel Sentiment Corpus validation error")

    test_csv = Path(data_home).expanduser() / "hotel_sentiment"
    test_csv = test_csv / "hotel_sentiment_target.csv.gz"

    with gzip.open(test_csv, "rt") as csvfile:
        rd = csv.DictReader(csvfile)

        for rec in rd:
            yield dict(id=rec['text_uuid'], sentiment_class=int(rec['sentiment']))
