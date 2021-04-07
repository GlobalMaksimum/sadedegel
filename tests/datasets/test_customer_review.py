import pytest
from pathlib import Path  # pylint: disable=unused-import
from os.path import expanduser  # pylint: disable=unused-import

from sadedegel.dataset.customer_review_classification import load_customer_review_train
from sadedegel.dataset.customer_review_classification import load_customer_review_test
from sadedegel.dataset.customer_review_classification import load_customer_review_target
from sadedegel.dataset.customer_review_classification import CLASS_VALUES


__class_names__ = ['alisveris',
                   'anne-bebek',
                   'beyaz-esya',
                   'bilgisayar',
                   'cep-telefon-kategori',
                   'egitim',
                   'elektronik',
                   'emlak-ve-insaat',
                   'enerji',
                   'etkinlik-ve-organizasyon',
                   'finans',
                   'gida',
                   'giyim',
                   'hizmet-sektoru',
                   'icecek',
                   'internet',
                   'kamu-hizmetleri',
                   'kargo-nakliyat',
                   'kisisel-bakim-ve-kozmetik',
                   'kucuk-ev-aletleri',
                   'medya',
                   'mekan-ve-eglence',
                   'mobilya-ev-tekstili',
                   'mucevher-saat-gozluk',
                   'mutfak-arac-gerec',
                   'otomotiv',
                   'saglik',
                   'sigortacilik',
                   'spor',
                   'temizlik',
                   'turizm',
                   'ulasim']


@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data/customer_review_classification")).exists()')
def test_data_load_train():
    data = load_customer_review_train()
    for i, row in enumerate(data):
        assert any(key in row.keys() for key in ['id', 'text', 'review_class'])
        assert isinstance(row['id'], str)
        assert isinstance(row['text'], str)
        assert CLASS_VALUES[row['review_class']] in __class_names__
    assert i + 1 == 323479


@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data/customer_review_classification")).exists()')
def test_data_load_test():
    data = load_customer_review_test()
    for i, row in enumerate(data):
        assert any(key in row.keys() for key in ['id', 'tweet'])
        assert isinstance(row['id'], str)
        assert isinstance(row['text'], str)
    assert i + 1 == 107827


@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data/customer_review_classification")).exists()')
def test_data_load_target():
    data = load_customer_review_target()
    for i, row in enumerate(data):
        assert any(key in row.keys() for key in ['id', 'sentiment_class'])
        assert isinstance(row['id'], str)
        assert CLASS_VALUES[row['review_class']] in __class_names__
    assert i + 1 == 107827
