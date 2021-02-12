from typing import List
import numpy as np
from rich.console import Console

from ..tokenize import Sentences
from .trainable.lda import load, get_topic_vector
from ._base import ExtractiveSummarizer

console = Console()


class LDASummarizer(ExtractiveSummarizer):
    tags = ExtractiveSummarizer.tags + ["unsupervised", "topic", "trainable"]

    def __init__(self, normalize=True):
        super().__init__(normalize)
        self.model = load()
        self.document_topic_ix = None

    def _lda_scorer(self, text: str):
        score = self.model.transform([text])[0][self.document_topic_ix]
        return score

    def _predict(self, sents: List[Sentences]):
        if self.model is not None:
            self.model['tfidf'].show_progress = False

        self.document_topic_ix = np.argmax(self.model.transform([self.document.raw]))

        return np.array([self._lda_scorer(sent.text) for sent in sents])
