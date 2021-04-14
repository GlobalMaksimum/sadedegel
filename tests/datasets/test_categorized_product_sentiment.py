import pytest
from .context import load_categorized_product_sentiment_train, SENTIMENT_CLASS_VALUES, PRODUCT_CATEGORIES


@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data/categorized_product_sentiment")).exists()')
def test_data_load():
    data = load_categorized_product_sentiment_train()

    count = 0
    for row in data:
        assert any(key in row.keys() for key in ['id', 'text', 'sentiment_class', 'product_category'])
        assert isinstance(row['id'], str)
        assert isinstance(row['text'], str)

        count += 1

    assert count == 5600


@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data/categorized_product_sentiment")).exists()')
@pytest.mark.parametrize('subset', ['Kitchen', 'DVD', 'Books', 'Electronics'])
def test_data_subset_str(subset):
    data = load_categorized_product_sentiment_train(categories=subset)
    count = 0
    for row in data:
        assert any(key in row.keys() for key in ['id', 'text', 'sentiment_class', 'product_category'])
        assert isinstance(row['id'], str)
        assert isinstance(row['text'], str)
        assert PRODUCT_CATEGORIES[row['product_category']] == subset

        count += 1

    assert count == 1400


@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data/categorized_product_sentiment")).exists()')
def test_data_subset_list():
    lov = ['Kitchen', 'DVD']
    data = load_categorized_product_sentiment_train(categories=lov)
    count = 0
    for row in data:
        assert any(key in row.keys() for key in ['id', 'text', 'sentiment_class', 'product_category'])
        assert isinstance(row['id'], str)
        assert isinstance(row['text'], str)
        assert PRODUCT_CATEGORIES[row['product_category']] in lov

        count += 1

    assert count == 2800
