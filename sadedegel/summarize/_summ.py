from math import ceil


class FirstK:
    def __init__(self, k=3):
        self.k = k

    def _select(self, sents):

        if type(self.k) == int:
            limit = self.k
        else:
            limit = min(ceil(self.k * len(sents)), len(sents))

        for i, sent in enumerate(sents):
            if i < limit:
                yield sent

    def __call__(self, sents, return_iter=False):
        if return_iter:
            return self._select(sents)
        else:
            return list(self._select(sents))
