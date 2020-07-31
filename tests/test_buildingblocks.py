import numpy as np
import torch # noqa # pylint: disable=import-error
from .context import Doc


def test_tokens():
    d = Doc("Ali topu tut. Ömer ılık süt iç.")

    s0 = d.sents[0]

    assert s0.tokens == ['Ali', 'topu', 'tut', '.']
    assert s0.tokens_with_special_symbols == ['[CLS]', 'Ali', 'topu', 'tut', '.', '[SEP]']


def test_padded_matrix():
    d = Doc("Ali topu tut. Ömer ılık süt iç.")

    assert np.all(d.padded_matrix(True, False) == np.array([[2, 3744, 9290, 2535, 18, 3, 0],
                                                            [2, 6565, 17626, 5244, 2032, 18, 3]]))

    assert torch.all(d.padded_matrix(return_mask=False) == torch.tensor([[2, 3744, 9290, 2535, 18, 3, 0],   # noqa # pylint: disable=not-callable
                                                                         [2, 6565, 17626, 5244, 2032, 18, 3]]))
