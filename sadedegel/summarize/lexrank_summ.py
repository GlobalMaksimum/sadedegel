import numpy as np
from lexrank import LexRank
from ..dataset import load_stopwords, load_raw_corpus
from ._base import ExtractiveSummarizer

class LexRankSummarizer(ExtractiveSummarizer):
    """
        Unsupervised summarizer which just like Rouge1 summarizer
        uses sentence's similarity to other sentences.
        Github: https://github.com/crabcamp/lexrank/
    """

    def __init__(self, normalize=True):
        self.normalize = normalize

        stopwords = set(load_stopwords())
        corpus = load_raw_corpus(return_iter=False)
        self.lxr = LexRank(corpus, stopwords=stopwords)


    def predict(self, sentences):
        sentences = [str(s) for s in sentences]
        scores = self.lxr.rank_sentences(
            sentences,
            threshold=None,
            fast_power_method=False,
        )

        scores = np.array(scores)
        if self.normalize:
            return scores / scores.sum()
        else:
            return scores
