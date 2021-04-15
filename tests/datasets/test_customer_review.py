import pytest
from pathlib import Path  # pylint: disable=unused-import
from os.path import expanduser  # pylint: disable=unused-import

from sadedegel.dataset.customer_review import load_train
from sadedegel.dataset.customer_review import load_test
from sadedegel.dataset.customer_review import load_test_label
from sadedegel.dataset.customer_review import CLASS_VALUES


@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data/customer_review_classification")).exists()')
def test_data_load_train():
    data = load_train()

    count = 0
    for row in data:
        assert any(key in row.keys() for key in ['id', 'text', 'review_class'])
        assert isinstance(row['id'], str)
        assert isinstance(row['text'], str)
        assert 0 <= row['review_class'] < len(CLASS_VALUES)

        count += 1

    assert count == 323479


@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data/customer_review_classification")).exists()')
def test_data_load_test():
    data = load_test()

    count = 0
    for row in data:
        assert any(key in row.keys() for key in ['id', 'tweet'])
        assert isinstance(row['id'], str)
        assert isinstance(row['text'], str)

        count += 1
    assert count == 107827


@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data/customer_review_classification")).exists()')
def test_data_load_target():
    data = load_test_label()
    count = 0
    for row in data:
        assert any(key in row.keys() for key in ['id', 'sentiment_class'])
        assert isinstance(row['id'], str)
        assert 0 <= row['review_class'] < len(CLASS_VALUES)

        count += 1
    assert count == 107827
