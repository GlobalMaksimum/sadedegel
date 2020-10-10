from typing import List
import numpy as np
from ._base import ExtractiveSummarizer
from ..bblock import Sentences


class RandomSummarizer(ExtractiveSummarizer):
    """Assign a random importance score to each sentences.

    seed: int, optional(default=42)
        Random generator seed.

    normalize : bool, optional (default=True)
        If ``False``, return a raw score vector.
        Otherwise, return a score vector that adds up to 1.
    """

    tags = ExtractiveSummarizer.tags + ['baseline']

    def __init__(self, seed=42, normalize=True):
        super().__init__(normalize)
        self.seed = seed
        np.random.seed(self.seed)

    def _predict(self, sentences: List[Sentences]) -> np.ndarray:
        if type(sentences) == list:
            n = len(sentences)
        else:
            n = len(list(sentences))

        return np.random.random(n)


class LengthSummarizer(ExtractiveSummarizer):
    """Assign a higher importance score to longer sentences.

    mode : {'token', 'char'}, default='token'
        Whether the distance should be defined by total number of characters or
        total number of tokens in a sentence.

    normalize : bool, optional (default=True)
        If ``False``, return a raw score vector.
        Otherwise, return a score vector that adds up to 1.
    """

    tags = ExtractiveSummarizer.tags + ['baseline']

    def __init__(self, mode="token", normalize=True):
        super().__init__(normalize)

        if mode not in ['token', 'char']:
            raise ValueError(f"mode should be one of 'token', 'char'")

        self.mode = mode

    def _predict(self, sentences: List[Sentences]) -> np.ndarray:

        if self.mode == 'token':
            scores = np.array([len(sent.tokens) for sent in sentences])
        else:
            scores = np.array([sum(len(token) for token in sent.tokens) for sent in sentences])

        return scores


class PositionSummarizer(ExtractiveSummarizer):
    """Assign a higher importance score based on relative position in documentation.

    mode : {'first', 'last'}, default='first'
        Whether the smaller position indices have higher scores.

    normalize : bool, optional (default=True)
        If ``False``, return a raw score vector.
        Otherwise, return a score vector that adds up to 1.
    """

    tags = ExtractiveSummarizer.tags + ['baseline']

    def __init__(self, mode='first', normalize=True):
        super().__init__(normalize)

        if mode not in ['first', 'last']:
            raise ValueError(f"mode should be one of 'first', 'last'")

        self.mode = mode

    def _predict(self, sentences: List[Sentences]) -> np.ndarray:
        if type(sentences) == list:
            n = len(sentences)
        else:
            n = len(list(sentences))

        if self.mode == "first":
            scores = np.arange(n - 1, -1, -1)
        else:
            scores = np.arange(n)

        return scores


class BandSummarizer(ExtractiveSummarizer):
    """Split document into bands of length k.

    Scoring scheme has two parts:
        1. Each band is scored by PositionSummarizer individually.
        2. Relative score for the same relative position within the band is also define by `mode` parameter.

    mode : {'forward', 'backward'}, default='forward'
        Whether the smaller position indices within a band have higher scores.

    k : int, default=3
        Split document into bands of length k sentences (except the last one).

    normalize : bool, optional (default=True)
        If ``False``, return a raw score vector.
        Otherwise, return a score vector that adds up to 1.
    """

    tags = ExtractiveSummarizer.tags + ['baseline']

    def __init__(self, k=3, mode='forward', normalize=True):
        super().__init__(normalize)
        self.k = k

        if mode not in ['forward', 'backward']:
            raise ValueError(f"mode should be one of 'forward', 'backward'")
        elif mode == 'backward':
            raise NotImplementedError(f"mode='backward' is not implemented yet.")

        self.mode = mode

    def _predict(self, sentences: List[Sentences]) -> np.ndarray:
        r = 0
        j = 0
        n = len(sentences)
        scores = [0 for _ in range(n)]

        while j < self.k:
            i = j

            while i < n:
                scores[i] = r
                r += 1
                i += self.k

            j += 1

        return (n - 1) - np.array(scores)
