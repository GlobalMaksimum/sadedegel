import pytest
from .context import load_categorized_product_sentiment_train, SENTIMENT_CLASS_VALUES, PRODUCT_CLASS_VALUES


def test_data_load():
    data = load_categorized_product_sentiment_train()

    count = 0
    for row in data:
        assert any(key in row.keys() for key in ['id', 'text', 'sentiment_class', 'product_category'])
        assert isinstance(row['id'], str)
        assert isinstance(row['text'], str)

        count += 1

    assert count == 5600


@pytest.mark.parametrize('subset', ['Kitchen', 'DVD', 'Books', 'Electronics'])
def test_data_subset(subset):

    data = load_categorized_product_sentiment_train(subset)
    count = 0
    for row in data:
        assert any(key in row.keys() for key in ['id', 'text', 'sentiment_class', 'product_category'])
        assert isinstance(row['id'], str)
        assert isinstance(row['text'], str)
        assert  PRODUCT_CLASS_VALUES[row['product_category']]  == subset

        count += 1

    assert count == 1400