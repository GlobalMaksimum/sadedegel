import csv
import gzip
from pathlib import Path
from rich.console import Console

from typing import Union, List, Iterator

SENTIMENT_CLASS_VALUES = ['POSITIVE', 'NEGATIVE']
PRODUCT_CATEGORIES = ['Kitchen', 'DVD', 'Books', 'Electronics']
CORPUS_SIZE = 5600

console = Console()

__general_download_message__ = """Ensure that you have properly downloaded categorized product sentiment corpus using

            python -m sadedegel.dataset.tweet_sentiment download --access-key xxx --secret-key xxxx

        Unfortunately due to data licensing issues we could not share data publicly. 
        Get in touch with sadedegel team to obtain a download key.
        """


def check_directory_structure(path: str) -> bool:
    base_dir = Path(path).expanduser()

    cat_prod_sentiment_dir = base_dir / 'categorized_product_sentiment'

    if not base_dir.exists():
        console.log(f"Dataset base directory ([bold red]{base_dir}[/bold red]) does not exist")

    elif not cat_prod_sentiment_dir.exists():
        console.log(
            f"Tweet sentiment directory ([bold red]{cat_prod_sentiment_dir}[/bold red]) does not exist")

    else:
        return True

    console.log(__general_download_message__)

    return False


def load_categorized_product_sentiment_train(data_home="~/.sadedegel_data",
                                             categories: Union[None, List[str], str] = None) -> Iterator[dict]:
    """

    @param data_home: Sadedegel data directory base. Default to be ~/.sadedegel_data
    @param categories:
        If None (default), load all the categories.
        If not None, list of category names (or a single category) to load (other categories
        ignored).
    @return: Iterator of dictionary
    """

    if not check_directory_structure(data_home):
        raise Exception("Categorized Product Corpus validation error")

    train_csv = Path(data_home).expanduser() / "categorized_product_sentiment"
    train_csv = train_csv / "categorized_product_sentiment.csv.gz"

    if categories is None:
        filtered_categories = PRODUCT_CATEGORIES
    elif isinstance(categories, str):
        filtered_categories = [categories]
    elif isinstance(categories, list):
        filtered_categories = categories
    else:
        raise ValueError(f"categories of type {type(categories)} is invalid.")

    with gzip.open(train_csv, "rt") as csvfile:
        rd = csv.DictReader(csvfile)

        for rec in rd:
            d = dict(id=rec['text_uuid'], text=rec['text'],
                     product_category=PRODUCT_CATEGORIES.index(rec['category']),
                     sentiment_class=SENTIMENT_CLASS_VALUES.index(rec['sentiment_class']))

            if rec['category'] in filtered_categories:
                yield d
