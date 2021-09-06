import sys
from pathlib import Path

sys.path.insert(0, (Path(__file__) / '..' / '..').absolute())

from sadedegel.dataset import load_raw_corpus  # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.prebuilt import news_classification, tweet_profanity, telco_sentiment  # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.dataset.tscorpus import CATEGORIES # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.dataset.profanity import CLASS_VALUES # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.prebuilt import tweet_sentiment , movie_reviews, customer_reviews_classification, hotel_sentiment,food_reviews # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.dataset.tweet_sentiment import CLASS_VALUES as SENTIMENT_VALUES  # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.dataset.movie_sentiment import CLASS_VALUES as SENTIMENT_VALUES_M  # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.dataset.telco_sentiment import CLASS_VALUES as SENTIMENT_VALUES_T  # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.dataset.customer_review import CLASS_VALUES as CLASS_VALUES_CUST # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.dataset.hotel_sentiment import CLASS_VALUES as CLASS_VALUES_HOTEL # noqa # pylint: disable=unused-import, wrong-import-position
from sadedegel.dataset.food_review import CLASS_VALUES as CLASS_VALUES_FOOD # noqa # pylint: disable=unused-import, wrong-import-position
