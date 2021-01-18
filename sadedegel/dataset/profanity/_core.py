import csv
from pathlib import Path
from rich.console import Console

CLASS_VALUES = ['NOT', 'OFF']
CORPUS_SIZE = 31277

console = Console()

__general_download_message__ = """Ensure that you have properly downloaded profanity corpus using

            python -m sadedegel.dataset.profanity download --access-key xxx --secret-key xxxx
            
        Unfortunately due to data licensing issues we could not share data publicly. 
        Get in touch with sadedegel team to obtain a download key.
        """


def check_directory_structure(path: str) -> bool:
    base_dir = Path(path).expanduser()

    offenseval_dir = base_dir / 'profanity' / 'offenseval2020-turkish'

    if not base_dir.exists():
        console.log(f"Dataset base directory ([bold red]{base_dir}[/bold red]) does not exist")

    elif not offenseval_dir.exists():
        console.log(
            f"OffensEval directory ([bold red]{offenseval_dir}[/bold red]) does not exist")

    else:
        return True

    console.log(__general_download_message__)

    return False


def load_offenseval_train(data_home="~/.sadedegel_data"):
    if not check_directory_structure(data_home):
        raise Exception("Ts Corpus validation error")

    train_tsv = Path(data_home).expanduser() / "profanity" / "offenseval2020-turkish" / "offenseval-tr-training-v1"
    train_tsv = train_tsv / "offenseval-tr-training-v1.tsv"

    with open(train_tsv) as csvfile:
        rd = csv.DictReader(csvfile, delimiter='\t')

        for rec in rd:
            yield dict(id=int(rec['id']), tweet=rec['tweet'], profanity_class=CLASS_VALUES.index(rec['subtask_a']))


def load_offenseval_test(data_home="~/.sadedegel_data"):
    if not check_directory_structure(data_home):
        raise Exception("Ts Corpus validation error")

    test_tsv = Path(data_home).expanduser() / "profanity" / "offenseval2020-turkish" / "offenseval-tr-testset-v1"
    test_tsv = test_tsv / "offenseval-tr-testset-v1.tsv"

    with open(test_tsv) as csvfile:
        rd = csv.DictReader(csvfile, delimiter='\t')

        for rec in rd:
            yield dict(id=int(rec['id']), tweet=rec['tweet'])


def load_offenseval_test_label(data_home="~/.sadedegel_data"):
    if not check_directory_structure(data_home):
        raise Exception("Ts Corpus validation error")

    test_tsv = Path(data_home).expanduser() / "profanity" / "offenseval2020-turkish" / "offenseval-tr-testset-v1"
    test_tsv = test_tsv / "offenseval-tr-labela-v1.tsv"

    with open(test_tsv) as csvfile:
        rd = csv.DictReader(csvfile, fieldnames=['id', 'profanity_class'])

        for rec in rd:
            yield dict(id=int(rec['id']), profanity_class=CLASS_VALUES.index(rec['profanity_class']))
