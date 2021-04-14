import csv
import gzip
from pathlib import Path
from rich.console import Console

SENTIMENT_CLASS_VALUES = ['POSITIVE', 'NEGATIVE']
PRODUCT_CLASS_VALUES = ['Kitchen', 'DVD', 'Books', 'Electronics']
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


def load_categorized_product_sentiment_train(subset=None, data_home="~/.sadedegel_data"):
    """

    :param subset: Subset of product category. default=None. Valid Values: ['Kitchen', 'DVD', 'Books', 'Electronics']
    :param data_home:
    :return: iter(dict)
    """
    if not check_directory_structure(data_home):
        raise Exception("Categorized Product Corpus validation error")

    train_csv = Path(data_home).expanduser() / "categorized_product_sentiment"
    train_csv = train_csv / "categorized_product_sentiment.csv.gz"

    with gzip.open(train_csv, "rt") as csvfile:
        rd = csv.DictReader(csvfile)

        if subset is None:
            for rec in rd:
                yield dict(id=rec['text_uuid'], text=rec['text'],
                           product_category=PRODUCT_CLASS_VALUES.index(rec['category']),
                           sentiment_class=SENTIMENT_CLASS_VALUES.index(rec['sentiment_class']))
        elif subset == 'Kitchen':
            for rec in rd:
                if rec['category'] == subset:
                    yield dict(id=rec['text_uuid'], text=rec['text'],
                               product_category=PRODUCT_CLASS_VALUES.index(rec['category']),
                               sentiment_class=SENTIMENT_CLASS_VALUES.index(rec['sentiment_class']))
        elif subset == 'Books':
            for rec in rd:
                if rec['category'] == subset:
                    yield dict(id=rec['text_uuid'], text=rec['text'],
                               product_category=PRODUCT_CLASS_VALUES.index(rec['category']),
                               sentiment_class=SENTIMENT_CLASS_VALUES.index(rec['sentiment_class']))
        elif subset == 'DVD':
            for rec in rd:
                if rec['category'] == subset:
                    yield dict(id=rec['text_uuid'], text=rec['text'],
                               product_category=PRODUCT_CLASS_VALUES.index(rec['category']),
                               sentiment_class=SENTIMENT_CLASS_VALUES.index(rec['sentiment_class']))
        elif subset == 'Electronics':
            for rec in rd:
                if rec['category'] == subset:
                    yield dict(id=rec['text_uuid'], text=rec['text'],
                               product_category=PRODUCT_CLASS_VALUES.index(rec['category']),
                               sentiment_class=SENTIMENT_CLASS_VALUES.index(rec['sentiment_class']))
        else:
            raise ValueError('Not a valid subset. Valid Values: ["Kitchen", "DVD", "Books", "Electronics"]')
