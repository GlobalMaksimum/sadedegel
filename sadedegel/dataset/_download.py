from zipfile import ZipFile
from smart_open import open
from pathlib import Path
from rich.console import Console
import os
import gzip
from shutil import copyfileobj


console = Console()

url_mapping = {
    "hotel_sentiment": "https://storage.googleapis.com/sadedegel/dataset/hotel_sentiment.zip",
    "categorized_prdoduct_sentiment": "https://storage.googleapis.com/sadedegel/dataset/categorized_product_sentiment.csv.gz",
    "movie_sentiment": "https://storage.googleapis.com/sadedegel/dataset/movie_sentiment.zip",
    "profanity": "https://storage.googleapis.com/sadedegel/dataset/offenseval2020-turkish.zip",
    "telco_sentiment": "https://storage.googleapis.com/sadedegel/dataset/telco_sentiment.zip",
    "tweet_sentiment": "https://storage.googleapis.com/sadedegel/dataset/tweet_sentiment_train.csv.gz",
    "product_sentiment": "https://storage.googleapis.com/sadedegel/dataset/product_sentiment.csv.gz",
    "customer_review": "https://storage.googleapis.com/sadedegel/dataset/customer_review_classification.zip",
    "food_review": "https://storage.googleapis.com/sadedegel/dataset/food_review.zip",
    "spam": "https://storage.googleapis.com/sadedegel/dataset/spam.csv.gz"
}

ts_corpus_tarballs = ["art_culture.jsonl.gz", "education.jsonl.gz", "horoscope.jsonl.gz", "life_food.jsonl.gz",
                      "politics.jsonl.gz", "technology.jsonl.gz", "economics.jsonl.gz", "health.jsonl.gz",
                      "life.jsonl.gz", "magazine.jsonl.gz", "sports.jsonl.gz", "travel.jsonl.gz"]

ts_corpus_fmts = ["raw", "tokenized"]


def form_directory(url: str, data_home="~/.sadedegel_data"):
    data_home = Path(os.path.expanduser(data_home))

    if "tscorpus" in url:
        data_home = data_home / 'tscorpus'
        for fmt in ts_corpus_fmts:
            (data_home / fmt).mkdir(parents=True, exist_ok=True)
            with console.status(f"[bold green]Downloading {fmt}...") as status:
                for tarball in ts_corpus_tarballs:
                    tb_url = f"{url}/{fmt}/{tarball}"
                    with open(tb_url, 'rb') as fp, open(data_home / fmt / tarball, "wb") as wp:
                        wp.write(fp.read())

                    console.log(f"{fmt}/{tarball} complete.")
    else:
        data_home.mkdir(parents=True, exist_ok=True)
        with console.status(f"[bold blue]Downloading {url.split('/')[-1]}"):
            if "gz" in url:
                with open(url, 'rb') as fp, gzip.open(data_home / os.path.basename(url), "wb") as wp:
                    copyfileobj(fp, wp)
            else:
                with open(url, 'rb') as fp:
                    with ZipFile(fp) as zp:
                        zp.extractall(data_home)


def download(dataset: str):
    if dataset in url_mapping.keys():
        url = url_mapping.get(dataset, None)
    elif dataset == "ts_corpus":
        url = "https://storage.googleapis.com/sadedegel/dataset/tscorpus"
    else:
        raise NotImplementedError(f"Dataset named {dataset} is not a part of sadedegel.dataset family. "
                                  f"Available datasets are: {list(url_mapping.keys()) + ['ts_corpus']}")
    form_directory(url)
