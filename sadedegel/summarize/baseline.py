import numpy as np
from ._base import ExtractiveSummarizer


class RandomSummarizer(ExtractiveSummarizer):
    """Assign a random importance score to each sentences.

    seed: int, optional(default=42)
        Random generator seed.

    normalize : bool, optional (default=True)
        If ``False``, return a raw score vector.
        Otherwise, return a score vector that adds up to 1.
    """

    def __init__(self, seed=42, normalize=True):
        self.seed = seed
        np.random.seed(self.seed)
        self.normalize = normalize

    def predict(self, sentences) -> np.ndarray:
        if type(sentences) == list:
            n = len(sentences)
        else:
            n = len(list(sentences))

        scores = np.random.random(n)

        if self.normalize:
            return scores / scores.sum()
        else:
            return scores


class LengthSummarizer(ExtractiveSummarizer):
    """Assign a higher importance score to longer sentences.

    mode : {'token', 'char'}, default='token'
        Whether the distance should be defined by total number of characters or
        total number of tokens in a sentence.

    normalize : bool, optional (default=True)
        If ``False``, return a raw score vector.
        Otherwise, return a score vector that adds up to 1.
    """

    def __init__(self, mode="token", normalize=True):
        self.normalize = normalize

        if mode not in ['token', 'char']:
            raise ValueError(f"mode should be one of 'token', 'char'")

        self.mode = mode

    def predict(self, sentences) -> np.ndarray:

        if self.mode == 'token':
            scores = np.array([len(sent.tokens) for sent in sentences])
        else:
            scores = np.array([sum(len(token) for token in sent.tokens) for sent in sentences])

        if self.normalize:
            return scores / scores.sum()
        else:
            return scores


class PositionSummarizer(ExtractiveSummarizer):
    """Assign a higher importance score based on relative position in documentation.

    mode : {'first', 'last'}, default='first'
        Whether the smaller position indices have higher scores.

    normalize : bool, optional (default=True)
        If ``False``, return a raw score vector.
        Otherwise, return a score vector that adds up to 1.
    """

    def __init__(self, mode='first', normalize=True):
        self.normalize = normalize
        if mode not in ['first', 'last']:
            raise ValueError(f"mode should be one of 'first', 'last'")

        self.mode = mode

    def predict(self, sentences) -> np.ndarray:
        if type(sentences) == list:
            n = len(sentences)
        else:
            n = len(list(sentences))

        if self.mode == "first":
            scores = np.arange(n - 1, -1, -1)
        else:
            scores = np.arange(n)

        if self.normalize:
            return scores / scores.sum()
        else:
            return scores


class BandSummarizer(ExtractiveSummarizer):
    """Split document into bands of length k.

    Scoring scheme has two parts:
        1. Each band is scored by PositionSummarizer individually.
        2. Relative score for the same relative position within the band is also define by `mode` parameter.

    mode : {'first', 'last'}, default='first'
        Whether the smaller position indices within a band have higher scores.

    k : int, default=3
        Split document into bands of length k sentences (except the last one).

    normalize : bool, optional (default=True)
        If ``False``, return a raw score vector.
        Otherwise, return a score vector that adds up to 1.
    """

    def __init__(self, mode='first', k=3, normalized=True):
        self.normalized = normalized
        self.k = k

        if mode not in ['first', 'last']:
            raise ValueError(f"mode should be one of 'first', 'last'")

        self.mode = mode

    def predict(self, sentences) -> np.ndarray:
        raise NotImplementedError("BandSummarizer is not completed yet.")
