from pathlib import Path  # pylint: disable=unused-import
from os.path import expanduser  # pylint: disable=unused-import

import pytest

from .context import hotel_sentiment


@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data/hotel_sentiment")).exists()')
def test_data_load():
    data = hotel_sentiment.load_hotel_sentiment_train()
    for i, row in enumerate(data):
        assert any(key in row.keys() for key in ['id', 'text', 'sentiment_class'])
        assert isinstance(row['id'], str)
        assert isinstance(row['text'], str)
        assert hotel_sentiment.CLASS_VALUES[row['sentiment_class']] in ['POSITIVE', 'NEGATIVE']
    assert i + 1 == 5800

@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data/hotel_sentiment")).exists()')
def test_data_load_test():
    data = hotel_sentiment.load_hotel_sentiment_test()
    for i, row in enumerate(data):
        assert any(key in row.keys() for key in ['id', 'text'])
        assert isinstance(row['id'], str)
        assert isinstance(row['text'], str)
    assert i +1 == 5800

@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data/hotel_sentiment")).exists()')
def test_data_load_label():
    data = hotel_sentiment.load_hotel_sentiment_test_label()
    for i, row in enumerate(data):
        assert any(key in row.keys() for key in ['id', 'sentiment_class'])
        assert isinstance(row['id'], str)
        assert hotel_sentiment.CLASS_VALUES[row['sentiment_class']] in ['POSITIVE', 'NEGATIVE']
    assert i +1 == 5800
