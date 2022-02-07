from rich.console import Console
from pathlib import Path
import gzip
import csv


TRAIN_SIZE, TEST_SIZE = 470396, 117599
CLASS_VALUES = ["NEGATIVE", "POSITIVE"]

console = Console()

__general_download_message__ = """Ensure that you have properly downloaded food review corpus using

            python -m sadedegel.dataset.food_review download --access-key xxx --secret-key xxxx

        Unfortunately due to data licensing issues we could not share data publicly. 
        Get in touch with sadedegel team to obtain a download key.
        """


def check_directory_structure(path: str) -> bool:
    base_dir = Path(path).expanduser()

    dataset_dir = base_dir / 'food_review'

    if not base_dir.exists():
        console.log(f"Dataset base directory ([bold red]{base_dir}[/bold red]) does not exist")

    elif not dataset_dir.exists():
        console.log(
            f" directory ([bold red]{dataset_dir}[/bold red]) does not exist")

    else:
        return True

    console.log(__general_download_message__)

    return False


def load_food_review_train(data_home="~/.sadedegel_data"):
    if not check_directory_structure(data_home):
        raise Exception(" validation error")

    train_csv = Path(data_home).expanduser() / "food_review"
    train_csv = train_csv / "food_reviews_train.csv.gz"

    with gzip.open(train_csv, "rt") as csvfile:
        rd = csv.DictReader(csvfile)

        for rec in rd:
            yield dict(id=rec['text_uuid'],
                       text=rec['review'],
                       sentiment_class=int(rec['sentiment_class']),
                       speed=int(rec['speed']),
                       service=int(rec['service']),
                       flavour=int(rec['flavour']))


def load_food_review_test(data_home="~/.sadedegel_data"):
    if not check_directory_structure(data_home):
        raise Exception(" validation error")

    train_csv = Path(data_home).expanduser() / "food_review"
    train_csv = train_csv / "food_reviews_test.csv.gz"

    with gzip.open(train_csv, "rt") as csvfile:
        rd = csv.DictReader(csvfile)

        for rec in rd:
            yield dict(id=rec['text_uuid'],
                       text=rec['review'],
                       sentiment_class=int(rec['sentiment_class']),
                       speed=int(rec['speed']),
                       service=int(rec['service']),
                       flavour=int(rec['flavour']))
