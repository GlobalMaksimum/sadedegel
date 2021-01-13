import sys
from pathlib import Path

sys.path.insert(0, (Path(__file__) / '..' / '..').absolute())

from sadedegel.dataset import load_raw_corpus  # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.prebuilt import news_classification, customer_sentiment  # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.dataset.tscorpus import CATEGORIES # noqa # pylint: disable=unused-import, wrong-import-position
