import pytest
from pathlib import Path  # pylint: disable=unused-import
from os.path import expanduser  # pylint: disable=unused-import
from .context import load_product_sentiment_train, PS_CLASS_VALUES


@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data/product_sentiment")).exists()')
def test_data_load():
    data = load_product_sentiment_train()
    for i, row in enumerate(data):
        assert any(key in row.keys() for key in ['text', 'sentiment_class'])
        assert isinstance(row['text'], str)
        assert PS_CLASS_VALUES[row['sentiment_class']] in ['NEUTRAL', 'POSITIVE', 'NEGATIVE']
    assert i + 1 == 11426
