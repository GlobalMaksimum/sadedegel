import csv
import gzip
from pathlib import Path
from rich.console import Console

CLASS_VALUES = ['Normal', 'Spam']
CORPUS_SIZE = 4751

console = Console()


__general_download_message__ = """Ensure that you have properly downloaded spam SMS corpus using
        sadedegel.dataset.download API
        """


def check_directory_structure(path: str) -> bool:
    base_dir = Path(path).expanduser()

    spam_sms_dir = base_dir / 'spam'

    if not base_dir.exists():
        console.log(f"Dataset base directory ([bold red]{base_dir}[/bold red]) does not exist")

    elif not spam_sms_dir.exists():
        console.log(
            f"Tweet sentiment directory ([bold red]{spam_sms_dir}[/bold red]) does not exist")

    else:
        return True

    console.log(__general_download_message__)

    return False


def load_spam_corpus(data_home="~/.sadedegel_data"):
    if not check_directory_structure(data_home):
        raise Exception("Tweet Sentiment Corpus validation error")

    train_csv = Path(data_home).expanduser() / "spam"
    train_csv = train_csv / "spam.csv.gz"

    with gzip.open(train_csv, "rt") as csvfile:
        rd = csv.DictReader(csvfile)

        for rec in rd:
            yield dict(text=rec['text'], spam_class=rec['class'])
