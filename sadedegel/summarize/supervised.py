from os.path import dirname
from pathlib import Path

import numpy as np
from typing import List
import joblib
from rich.console import Console

from ._base import ExtractiveSummarizer
from ..bblock.util import __transformer_model_mapper__
from ..bblock import Sentences
from ..bblock.doc import DocBuilder
from ..config import load_config


__vector_types__ = list(__transformer_model_mapper__.keys()) + ["tfidf", "bm25"]
console = Console()


def load_model(vector_type):
    name = f"ranker_{vector_type}.joblib"
    path = (Path(dirname(__file__)) / 'model' / name).absolute()
    console.log(f"Initializing ranker model ranker_{vector_type}...", style="blue")

    try:
        model = joblib.load(path)
    except Exception as e:
        raise FileNotFoundError(f"A model trained for {vector_type} is not found. Please optimize one with "
                                f"sadedegel.summarize.RankerOptimizer.")
    return model


class SupervisedSentenceRanker(ExtractiveSummarizer):
    model = None
    vector_type = None
    tags = ExtractiveSummarizer.tags + ["ml", "supervised", "rank"]

    def __init__(self, normalize=True, vector_type="bert_128k_cased"):
        super().__init__(normalize)
        self.init_model(vector_type)

    @classmethod
    def init_model(cls, vector_type):
        if vector_type not in __vector_types__:
            raise ValueError(f"Not a valid vectorization for input sequence. Valid types are {__vector_types__}")
        if cls.vector_type is not None:
            if cls.vector_type == vector_type:
                return 0

        cls.model = load_model(vector_type)
        cls.vector_type = vector_type

    def _predict(self, sents: List[Sentences]) -> np.ndarray:
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


class RankerOptimizer:
    def __init__(self, vector_type: str, summarization_perc: float,**kwargs):
        self.vector_type = vector_type
        self.summarization_perc = summarization_perc


    @property
    def optimize(self):
        """Optimize the ranker model for a custom summarization percentage. Optimize and dump a new model.

        Parameters
        ----------
        summarization_perc: float
            Percentage of summary to optimize for. Range: [0, 1]
        """
        pass

    def _prepare_dataset(self):
        pass
