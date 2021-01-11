from typing import List, Union
import numpy as np  # type: ignore
from abc import ABC, abstractmethod
from ..bblock import Sentences
from ..bblock.doc import DocBuilder, Document
import warnings


def get_sentences_list(X: Union[Document, List[Sentences], List[str]]):
    if type(X) == Document:
        sentences = list(X)  # Get the list of Sentences from Doc
    elif type(X) != list:
        raise ValueError(f"sents parameter should be one of Doc, List[Sentences] or List[str]. Found {type(X)}")
    elif all(type(s) == str for s in X):
        d = DocBuilder().from_sentences(X)
        sentences = list(d)
    elif all(type(s) == Sentences for s in X):
        sentences = X
    else:
        raise ValueError(f"sents parameter should be one of Doc, List[Sentences] or List[str]. Found {type(X)}")

    return sentences


class ExtractiveSummarizer(ABC):
    tags = ['extractive']

    def __init__(self, normalize=True):
        self.normalize = normalize

    @abstractmethod
    def _predict(self, sents: List[Sentences]) -> np.ndarray:
        pass

    def predict(self, sents: Union[Document, List[Sentences], List[str]]) -> np.ndarray:
        """Predict relevance score for X

        Parameters
        ----------
        sents:
          Doc: sadedeGel Document
          List[Sentences]: List of sadedegel Sentences
          List[str]: List of strings (each element is taken to a sentences)

        Returns
        -------
        Relevance score per sentence
        np.array
        """
        sents = get_sentences_list(sents)

        scores = self._predict(sents)

        if self.normalize:
            return scores / scores.sum()
        else:
            return scores

    def __call__(self, sents: Union[Document, List[Sentences], List[str]], k: int) -> List[Sentences]:

        sents = get_sentences_list(sents)

        if k > len(sents):
            warnings.warn(f"k ({k}) is greater then the number of sentences ({len(sents)})", UserWarning)
            k = len(sents)
        elif k == 0:
            return []

        scores = self.predict(sents)

        topk_inds = np.argpartition(scores, k - 1)[-k:]  # returns indices of k top sentences
        topk_inds.sort()  # fix the order of sentences

        summ = [sents[i] for i in topk_inds]

        return summ

    def __contains__(self, tag: str) -> bool:
        """Check whether instance tags contain a given tag for filtering summarizers.
        """

        return tag in self.tags
