import pytest
from pytest import warns, raises
from .context import TextRank, Doc
import numpy as np


long_text = "Merhaba dünya biz dostuz. Barış için geldik. Sizi lazerlerimizle buharlaştırmayacağız. " \
           "Onun yerine kölemiz olacaksınız."


def val_err(method, d):
    method(d, k=1)


def sent_err(method, sents):
    method.predict(sents)


@pytest.mark.parametrize("input_type, normalize, text",
                         [pytest.param('bert', True, long_text, id='bert'),
                          pytest.param('tfidf', True, long_text, id='nonexistent'),
                          pytest.param('bert', False, long_text, id='norm_false'),
                          pytest.param('bert', True, long_text, id='norm_true'),
                          pytest.param('bert', True, [], id='no text')])
def test_text_rank(input_type, normalize, text):
    if not text:
        with raises(ValueError, match=r"Ensure that document .*"):
            sent_err(TextRank(input_type), text)
    else:
        d = Doc(text)
        if input_type != 'bert':
            with raises(ValueError, match=r"mode should be one of .*"):
                val_err(TextRank(input_type), d)
        if input_type == 'bert':
            with warns(UserWarning, match="Changing tokenizer to"):
                assert len(TextRank(input_type)(d, k=1)) == 1
                if normalize:
                    pred = TextRank(input_type, alpha=0.5, normalize=normalize).predict(d)
                    assert np.float16(pred[0]) == np.float16(0.25474593)
                else:
                    pred = TextRank(input_type, alpha=0.5, normalize=normalize).predict(d)
                    assert np.float16(pred[0]) == np.float16(0.25474593)



