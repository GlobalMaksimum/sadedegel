import csv
from pathlib import Path
from rich.console import Console
import gzip

CLASS_VALUES = ['Alışveriş', 'Anne-Bebek', 'Beyaz-Eşya', 'Bilgisayar', 'Cep Telefon Kategori', 'Eğitim', 'Elektronik',
                'Emlak ve İnşaat', 'Enerji', 'Etkinlik ve Organizasyon', 'Finans',
                'Gıda', 'Giyim', 'Hizmet Sektörü', 'İçecek', 'İnternet', 'Kamu Hizmetleri', 'Kargo Nakliyat',
                'Kişisel Bakım ve Kozmetik', 'Küçük Ev Aletleri', 'Medya', 'Mekan ve Eğlence', 'Mobilya Ev Tekstili',
                'Mücevher Saat Gözlük', 'Mutfak Araç Gereç', 'Otomotiv', 'Sağlık',
                'Sigortacılık', 'Spor', 'Temizlik',
                'Turizm', 'Ulaşım']

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


def load_train(data_home="~/.sadedegel_data"):
    if not check_directory_structure(data_home):
        raise Exception("Customer Review Classification Corpus validation error")

    train_csv = Path(data_home).expanduser() / "customer_review_classification"
    train_csv = train_csv / "customer_review_train.csv.gz"

    with gzip.open(train_csv, "rt") as csvfile:
        rd = csv.DictReader(csvfile)

        for rec in rd:
            yield dict(id=rec['text_uuid'], text=rec['text'], review_class=int(rec['review_class']))


def load_test(data_home="~/.sadedegel_data"):
    if not check_directory_structure(data_home):
        raise Exception("Customer Review Classification Corpus validation error")

    test_csv = Path(data_home).expanduser() / "customer_review_classification"
    test_csv = test_csv / "customer_review_test.csv.gz"

    with gzip.open(test_csv, "rt") as csvfile:
        rd = csv.DictReader(csvfile)

        for rec in rd:
            yield dict(id=rec['text_uuid'], text=rec['text'])


def load_test_label(data_home="~/.sadedegel_data"):
    if not check_directory_structure(data_home):
        raise Exception("Customer Review Classification Corpus validation error")

    target_csv = Path(data_home).expanduser() / "customer_review_classification"
    target_csv = target_csv / "customer_review_target.csv.gz"

    with gzip.open(target_csv, "rt") as csvfile:
        rd = csv.DictReader(csvfile)

        for rec in rd:
            yield dict(id=rec['text_uuid'], review_class=int(rec['review_class']))
