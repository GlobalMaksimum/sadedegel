from pathlib import Path  # pylint: disable=unused-import
from os.path import expanduser  # pylint: disable=unused-import

import pytest

from .context import load_tweet_sentiment_train, CLASS_VALUES


@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data/tweet_sentiment")).exists()')
def test_data_load():
    data = load_tweet_sentiment_train()
    for i, row in enumerate(data):
        assert any(key in row.keys() for key in ['id', 'tweet', 'sentiment_class'])
        assert isinstance(row['id'], str)
        assert isinstance(row['tweet'], str)
        assert CLASS_VALUES[row['sentiment_class']] in ['POSITIVE', 'NEGATIVE']
    assert i == 11116
