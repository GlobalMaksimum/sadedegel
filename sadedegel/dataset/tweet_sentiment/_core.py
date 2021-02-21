import csv
import gzip
from pathlib import Path
from rich.console import Console

CLASS_VALUES = ['POSITIVE', 'NEGATIVE']
CORPUS_SIZE = 11117

console = Console()

__general_download_message__ = """Ensure that you have properly downloaded tweet sentiment corpus using

            python -m sadedegel.dataset.tweet_sentiment download --access-key xxx --secret-key xxxx

        Unfortunately due to data licensing issues we could not share data publicly. 
        Get in touch with sadedegel team to obtain a download key.
        """


def check_directory_structure(path: str) -> bool:
    base_dir = Path(path).expanduser()

    tweet_sentiment_dir = base_dir / 'tweet_sentiment'

    if not base_dir.exists():
        console.log(f"Dataset base directory ([bold red]{base_dir}[/bold red]) does not exist")

    elif not tweet_sentiment_dir.exists():
        console.log(
            f"Tweet sentiment directory ([bold red]{tweet_sentiment_dir}[/bold red]) does not exist")

    else:
        return True

    console.log(__general_download_message__)

    return False


def load_tweet_sentiment_train(data_home="~/.sadedegel_data"):
    if not check_directory_structure(data_home):
        raise Exception("Tweet Sentiment Corpus validation error")

    train_csv = Path(data_home).expanduser() / "tweet_sentiment"
    train_csv = train_csv / "tweet_sentiment_train.csv.gz"

    with gzip.open(train_csv, "rt") as csvfile:
        rd = csv.DictReader(csvfile)

        for rec in rd:
            yield dict(id=rec['text_uuid'], tweet=rec['text'], sentiment_class=CLASS_VALUES.index(rec['sentiment']))
