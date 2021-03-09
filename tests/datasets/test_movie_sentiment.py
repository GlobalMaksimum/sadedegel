from pathlib import Path  # pylint: disable=unused-import
from os.path import expanduser  # pylint: disable=unused-import

import pytest

from .context import movie_sentiment


@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data")).exists()')
def test_data_load():
    data = movie_sentiment.load_movie_sentiment_train()
    for i, row in enumerate(data):
        assert any(key in row.keys() for key in ['id', 'tweet', 'sentiment_class'])
        assert isinstance(row['id'], str)
        assert isinstance(row['text'], str)
        assert movie_sentiment.CLASS_VALUES[row['sentiment_class']] in ['POSITIVE', 'NEGATIVE']
    assert i + 1 == movie_sentiment.CORPUS_SIZE
