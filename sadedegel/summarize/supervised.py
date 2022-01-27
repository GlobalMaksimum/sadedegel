from os.path import dirname
from pathlib import Path

import numpy as np
from typing import List
import joblib

from ._base import ExtractiveSummarizer
from ..bblock.util import __transformer_model_mapper__
from ..bblock import Sentences
from ..bblock.doc import DocBuilder
from ..config import load_config


__vector_types__ = list(__transformer_model_mapper__.keys()) + ["tfidf", "bm25"]


def load_model(vector_type):
    name = f"ranker_{vector_type}.joblib"
    path = (Path(dirname(__file__)) / 'model' / name).absolute()
    return joblib.load(path)


class SupervisedSentenceRanker(ExtractiveSummarizer):
    def __init__(self, builder, normalize):
        super().__init__(normalize)
        self.model = builder.model
        self.vector_type = builder.vector_type

    def _predict(self, sents: List[Sentences]) -> np.ndarray:
        if self.vector_type not in __vector_types__:
            raise ValueError(f"Not a valid vectorization for input sequence. Valid types are {__vector_types__}")
        if self.vector_type not in ["tfidf", "bm25"]:
            doc_sent_embeddings = self._get_pretrained_embeddings(sents)
        else:
            raise NotImplementedError("BoW interface for SupervisedSentenceRanker is not yet implemented.")

        if self.model is not None:
            scores = self.model.predict(doc_sent_embeddings)
        else:
            raise ValueError("A ranker model is not found.")

        return scores

    def _get_pretrained_embeddings(self, sents: List[Sentences]) -> np.ndarray:
        doc_embedding = sents[0].document.get_pretrained_embedding(architecture=self.vector_type, do_sents=False)
        doc_embedding = np.vstack(len(sents) * [doc_embedding])
        sent_embeddings = sents[0].document.get_pretrained_embedding(architecture=self.vector_type, do_sents=True)

        return np.hstack([doc_embedding, sent_embeddings])

    def _get_bow_vectors(self, sents: List[Sentences]) -> np.ndarray:
        pass


class SupervisedSummarizer:
    def __init__(self, vector_type=None, **kwargs):
        if vector_type is None:  # Override config if user input is provided.
            self.vector_type = load_config(kwargs)["summarizer"]["vector_type"]
        else:
            self.vector_type = vector_type
        self.model = load_model(self.vector_type)

    def __call__(self):
        return SupervisedSentenceRanker(builder=self, normalize=True)

    def optimize(self, summarization_perc: float):
        """Optimize the ranker model for a custom summarization percentage. Optimize and dump a new model.

        Parameters
        ----------
        summarization_perc: float
            Percentage of summary to optimize for. Range: [0, 1]
        """
        pass
