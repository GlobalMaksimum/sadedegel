from pathlib import Path  # pylint: disable=unused-import
from os.path import expanduser  # pylint: disable=unused-import

import pytest

from .context import load_telco_sentiment_train, load_telco_sentiment_test, load_telco_sentiment_test_label
from .context import TELCO_CLASS_VALUES


@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data/telco_sentiment")).exists()')
def test_data_load():
    data = load_telco_sentiment_train()
    for i, row in enumerate(data):
        assert any(key in row.keys() for key in ['text_uuid', 'tweet', 'sentiment_class'])
        assert isinstance(row['id'], str)
        assert isinstance(row['tweet'], str)
        assert TELCO_CLASS_VALUES[row['sentiment_class']] in ['notr', 'olumlu', 'olumsuz']
    assert i + 1 == 13832


@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data/telco_sentiment")).exists()')
def test_data_load_test():
    data = load_telco_sentiment_test()
    for i, row in enumerate(data):
        assert any(key in row.keys() for key in ['text_uuid', 'tweet'])
        assert isinstance(row['id'], str)
        assert isinstance(row['tweet'], str)
    assert i + 1 == 3457


@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data/telco_sentiment")).exists()')
def test_data_load_target():
    data = load_telco_sentiment_test_label()
    for i, row in enumerate(data):
        assert any(key in row.keys() for key in ['id', 'sentiment_class'])
        assert isinstance(row['id'], str)
        assert TELCO_CLASS_VALUES[row['sentiment_class']] in ['notr', 'olumlu', 'olumsuz']
    assert i + 1 == 3457
