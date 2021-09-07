import sys
from pathlib import Path

sys.path.insert(0, (Path(__file__) / '..' / '..').absolute())

from sadedegel.dataset import load_raw_corpus, load_sentence_corpus,load_annotated_corpus # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.dataset.extended import load_extended_metadata, load_extended_sents_corpus, load_extended_raw_corpus  # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.dataset.tscorpus import load_tokenization_raw,load_tokenization_tokenized, check_and_display, CORPUS_SIZE # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.dataset.tweet_sentiment import load_tweet_sentiment_train, CLASS_VALUES # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.dataset.product_sentiment import load_product_sentiment_train # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.dataset.product_sentiment import CLASS_VALUES as PS_CLASS_VALUES # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.dataset.telco_sentiment import load_telco_sentiment_train, load_telco_sentiment_test, load_telco_sentiment_test_label # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.dataset.telco_sentiment import CLASS_VALUES as TELCO_CLASS_VALUES # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.dataset.categorized_product_sentiment import load_categorized_product_sentiment_train, SENTIMENT_CLASS_VALUES, PRODUCT_CATEGORIES # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.dataset import movie_sentiment # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.dataset import hotel_sentiment # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.bblock.cli.__main__ import tok_eval # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.dataset import util # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.dataset import file_paths, CorpusTypeEnum # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.dataset.food_review import load_food_review_train, load_food_review_test # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.dataset.food_review import CLASS_VALUES as FOOD_CLASS_VALUES # noqa # pylint: disable=unused-import, wrong-import-position
