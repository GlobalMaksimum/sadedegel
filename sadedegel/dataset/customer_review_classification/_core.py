import csv
from pathlib import Path
from rich.console import Console
import gzip

CLASS_VALUES = {0: 'alisveris',
 1: 'anne-bebek',
 2: 'beyaz-esya',
 3: 'bilgisayar',
 4: 'cep-telefon-kategori',
 5: 'egitim',
 6: 'elektronik',
 7: 'emlak-ve-insaat',
 8: 'enerji',
 9: 'etkinlik-ve-organizasyon',
 10: 'finans',
 11: 'gida',
 12: 'giyim',
 13: 'hizmet-sektoru',
 14: 'icecek',
 15: 'internet',
 16: 'kamu-hizmetleri',
 17: 'kargo-nakliyat',
 18: 'kisisel-bakim-ve-kozmetik',
 19: 'kucuk-ev-aletleri',
 20: 'medya',
 21: 'mekan-ve-eglence',
 22: 'mobilya-ev-tekstili',
 23: 'mucevher-saat-gozluk',
 24: 'mutfak-arac-gerec',
 25: 'otomotiv',
 26: 'saglik',
 27: 'sigortacilik',
 28: 'spor',
 29: 'temizlik',
 30: 'turizm',
 31: 'ulasim'}

CORPUS_SIZE = 323479

console = Console()

__general_download_message__ = """Ensure that you have properly downloaded customer review classification corpus using

            python -m sadedegel.dataset.customer_review_classification download --access-key xxx --secret-key xxxx

        Unfortunately due to data licensing issues we could not share data publicly. 
        Get in touch with sadedegel team to obtain a download key.
        """


def check_directory_structure(path: str) -> bool:
    base_dir = Path(path).expanduser()

    customer_review_dir = base_dir / 'customer_review_classification'

    if not base_dir.exists():
        console.log(f"Dataset base directory ([bold red]{base_dir}[/bold red]) does not exist")

    elif not customer_review_dir.exists():
        console.log(
            f"Customer revie classification dataset directory ([bold red]{customer_review_dir}[/bold red]) does not exist")

    else:
        return True

    console.log(__general_download_message__)

    return False


def load_customer_review_train(data_home="~/.sadedegel_data"):
    if not check_directory_structure(data_home):
        raise Exception("Customer Review Classification Corpus validation error")

    train_csv = Path(data_home).expanduser() / "customer_review_classification"
    train_csv = train_csv / "customer_review_train.csv.gz"

    with gzip.open(train_csv, "rt") as csvfile:
        rd = csv.DictReader(csvfile)

        for rec in rd:
            yield dict(id=rec['text_uuid'], text=rec['text'], review_class=int(rec['review_class']))


def load_customer_review_test(data_home="~/.sadedegel_data"):
    if not check_directory_structure(data_home):
        raise Exception("Customer Review Classification Corpus validation error")

    test_csv = Path(data_home).expanduser() / "customer_review_classification"
    test_csv = test_csv / "customer_review_test.csv.gz"

    with gzip.open(test_csv, "rt") as csvfile:
        rd = csv.DictReader(csvfile)

        for rec in rd:
            yield dict(id=rec['text_uuid'], text=rec['text'])


def load_customer_review_target(data_home="~/.sadedegel_data"):
    if not check_directory_structure(data_home):
        raise Exception("Customer Review Classification Corpus validation error")

    target_csv = Path(data_home).expanduser() / "customer_review_classification"
    target_csv = target_csv / "customer_review_target.csv.gz"

    with gzip.open(target_csv, "rt") as csvfile:
        rd = csv.DictReader(csvfile)

        for rec in rd:
            yield dict(id=rec['text_uuid'], review_class=int(rec['review_class']))
