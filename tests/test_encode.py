import pytest
import numpy as np
import scipy
from .context import encode, Token


__famous_quote__ = "Merhaba Dünya. Biz dostuz. Barış için geldik."


@pytest.mark.parametrize('emb_type, out_type', [('bert', np.ndarray),
                                                ('tfidf', scipy.sparse.csr_matrix),
                                                ('glove', ValueError)])
def test_encode_sent(emb_type, out_type):
    if emb_type == 'glove':
        with pytest.raises(out_type, match=r".*Not a valid embedding type .*"):
            encode(__famous_quote__, embed=emb_type, level='sentence')
    else:
        enc = encode(__famous_quote__, embed=emb_type, level='sentence')
        assert isinstance(enc, out_type)
        if emb_type == "tfidf":
            assert enc.shape == (3, len(Token.vocabulary()))
        elif emb_type == 'bert':
            assert enc.shape == (3, 768)


@pytest.mark.parametrize('emb_type', ["bert", "tfidf", "glove"])
def test_encode_doc(emb_type):
    if emb_type == "tfidf":
        enc = encode(__famous_quote__, embed=emb_type, level='document')
        assert enc.shape == (1, len(Token.vocabulary()))
    elif emb_type == "bert":
        with pytest.raises(NotImplementedError, match=r".*not in current release.*"):
            encode(__famous_quote__, embed=emb_type, level='document')
    elif emb_type == "glove":
        with pytest.raises(ValueError, match=r".*Not a valid embedding type .*"):
            encode(__famous_quote__, embed=emb_type, level='document')
