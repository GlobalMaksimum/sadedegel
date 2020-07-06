from math import ceil
from random import sample


class BasicSummarizer:
    def __call__(self, sents, return_iter=False):
        if return_iter:
            return self._select(sents)
        else:
            return list(self._select(sents))


class FirstK(BasicSummarizer):
    def __init__(self, k=3):
        self.k = k
        super().__init__()

    def _select(self, sents):

        if type(self.k) == int:
            limit = self.k
        else:
            limit = min(ceil(self.k * len(sents)), len(sents))

        for i, sent in enumerate(sents):
            if i < limit:
                yield sent


class RandomK(BasicSummarizer):
    def __init__(self, k=3):
        self.k = k
        super().__init__()

    def _select(self, sents):

        if type(self.k) == int:
            limit = self.k
        else:
            limit = min(ceil(self.k * len(sents)), len(sents))

        yield from sample(sents, limit)
