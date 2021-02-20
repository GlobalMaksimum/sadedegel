import sys
from pathlib import Path

sys.path.insert(0, (Path(__file__) / '..' / '..').absolute())

from sadedegel.dataset import load_raw_corpus  # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.dataset.extended import load_extended_raw_corpus  # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.extension.sklearn import  TfidfVectorizer, OnlinePipeline, BM25Vectorizer # noqa # pylint: disable=unused-import, wrong-import-position
