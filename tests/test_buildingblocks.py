import numpy as np
import torch
import pytest
from .context import Doc


def test_tokens():
    d = Doc("Ali topu tut. Ömer ılık süt iç.")

    s0 = d.sents[0]

    assert s0.tokens == ['Ali', 'topu', 'tut', '.']
    assert s0.tokens_with_special_symbols == ['[CLS]', 'Ali', 'topu', 'tut', '.', '[SEP]']


def test_bert_embedding_generation():
    d = Doc("Ali topu tut. Ömer ılık süt iç.")

    assert d.bert_embeddings.shape == (2, 768)


testdata = [(True, True),
            (True, False),
            (False, False),
            (False, True)]


@pytest.mark.parametrize("return_numpy, return_mask", testdata)
def test_padded_matrix(return_numpy, return_mask):
    d = Doc("Ali topu tut. Ömer ılık süt iç.")

    inp = np.array([[2, 3744, 9290, 2535, 18, 3, 0],
                    [2, 6565, 17626, 5244, 2032, 18, 3]])

    mask = np.array([[1, 1, 1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1, 1, 1]])

    res = d.padded_matrix(return_numpy, return_mask)

    if return_numpy:
        if return_mask:
            assert np.all(res[0] == inp)
            assert np.all(res[1] == mask)
        else:
            assert np.all(res == inp)
    else:
        if return_mask:
            assert torch.all(res[0] == torch.tensor(inp))  # noqa # pylint: disable=not-callable
            assert torch.all(res[1] == torch.tensor(mask))  # noqa # pylint: disable=not-callable
        else:
            assert torch.all(res == torch.tensor(inp))  # noqa # pylint: disable=not-callable


@pytest.mark.parametrize("test_for", ["text", "str", "strall"])
def test_span(test_for):
    d = Doc("Ali   topu  tut.")

    spans = d.spans

    if test_for == "text":
        assert [s.text for s in spans] == ['Ali', 'topu', 'tut.']

    elif test_for == "str":
        assert [str(s) for s in spans] == ['Ali', 'topu', 'tut.']
    else:
        assert str(spans) == "[Ali, topu, tut.]"
