import csv
import gzip
from pathlib import Path
from rich.console import Console

CLASS_VALUES = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
CORPUS_SIZE = 11426

console = Console()

__general_download_message__ = """Ensure that you have properly downloaded product sentiment corpus using

            python -m sadedegel.dataset.product_sentiment download --access-key xxx --secret-key xxxx

        Unfortunately due to data licensing issues we could not share data publicly. 
        Get in touch with sadedegel team to obtain a download key.
        """


def check_directory_structure(path: str) -> bool:
    base_dir = Path(path).expanduser()

    product_sentiment_dir = base_dir / 'product_sentiment'

    if not base_dir.exists():
        console.log(f"Dataset base directory ([bold red]{base_dir}[/bold red]) does not exist")

    elif not product_sentiment_dir.exists():
        console.log(
            f"Product sentiment directory ([bold red]{product_sentiment_dir}[/bold red]) does not exist")

    else:
        return True

    console.log(__general_download_message__)

    return False


def load_product_sentiment_train(data_home="~/.sadedegel_data"):
    if not check_directory_structure(data_home):
        raise Exception("Product Sentiment Corpus validation error")

    train_csv = Path(data_home).expanduser() / "product_sentiment"
    train_csv = train_csv / "product_sentiment.csv.gz"

    with gzip.open(train_csv, "rt") as csvfile:
        rd = csv.DictReader(csvfile)

        for rec in rd:
            yield dict(text=rec['text'], sentiment_class=CLASS_VALUES.index(rec['sentiment']))
