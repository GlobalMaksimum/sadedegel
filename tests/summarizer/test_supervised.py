import pytest
from pytest import raises
import numpy as np

from sklearn.exceptions import NotFittedError
from .context import SupervisedSummarizer, Doc


famous_quote = ("Merhaba dünya biz dostuz. Barış için geldik. Sizi lazerlerimizle buharlaştırmayacağız."
                "Onun yerine kölemiz olacaksınız.")


@pytest.mark.parametrize('model_type, text', [('rf.test', famous_quote),
                                              ('lgbm.test', famous_quote)])
def test_supervised(model_type, text):
    spv = SupervisedSummarizer(model_type=model_type)
    d = Doc(text)
    with raises(NotFittedError, match=r'.* not fitted.*'):
        spv.predict(d)
