from os.path import dirname
from pathlib import Path
from itertools import tee
import randomname

import numpy as np
from typing import List
import joblib
from rich.console import Console
from rich.progress import track

from ._base import ExtractiveSummarizer
from ..bblock.util import __transformer_model_mapper__
from ..bblock import Sentences
from ..bblock.doc import DocBuilder
from .util.supervised_tuning import optuna_handler, create_empty_model, fit_ranker, save_ranker


__vector_types__ = list(__transformer_model_mapper__.keys()) + ["tfidf", "bm25"]
console = Console()

try:
    import pandas as pd
except ImportError:
    console.log(("pandas package is not a general sadedegel dependency."
                 " But we do have a dependency on building our supervised ranker model"))


def load_model(vector_type, debug=False):
    name = f"ranker_{vector_type}.joblib"

    if vector_type == "bert_128k_cased":
        path = (Path(dirname(__file__)) / 'model' / name).absolute()
    else:
        path = Path(f"~/.sadedegel_data/models/{name}").expanduser()

    if not debug:
        try:
            model = joblib.load(path)
            console.log(f"Initializing ranker model ranker_{vector_type}...", style="blue")
        except Exception as e:
            raise FileNotFoundError(f"A model trained for {vector_type} is not found. Please optimize one with "
                                    f"sadedegel.summarize.RankerOptimizer. {e}")

    else:
        model = name

    return model


class SupervisedSentenceRanker(ExtractiveSummarizer):
    model = None
    vector_type = None
    debug = False
    tags = ExtractiveSummarizer.tags + ["ml", "supervised", "rank"]

    def __init__(self, normalize=True, vector_type="bert_128k_cased", **kwargs):
        super().__init__(normalize)
        self.debug = kwargs.get("debug", False)
        self.init_model(vector_type, self.debug)

    @classmethod
    def init_model(cls, vector_type, debug):
        db_switch = False
        if vector_type not in __vector_types__:
            raise ValueError(f"Not a valid vectorization for input sequence. Valid types are {__vector_types__}")
        if cls.debug != debug:
            cls.debug = debug
            db_switch = True
            if cls.debug:
                console.log("SupervisedSentenceRanker: Switching debug mode ON.")
            else:
                console.log("SupervisedSentenceRanker Switching debug mode OFF.")
        if cls.vector_type is not None and not db_switch:
            if cls.vector_type == vector_type:
                return 0

        cls.model = load_model(vector_type, debug)
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


class RankerOptimizer(SupervisedSentenceRanker):
    def __init__(self, n_trials: int, vector_type: str, summarization_perc: float,**kwargs):
        self.n_trials = n_trials
        self.vector_type = vector_type
        self.summarization_perc = summarization_perc

    def optimize(self):
        """Optimize the ranker model for a custom summarization percentage. Optimize and dump a new model.
        """
        run_name = randomname.get_name()
        df, vecs = self._prepare_dataset()

        optuna_handler(n_trials=self.n_trials, run_name=run_name,
                       metadata=df, vectors=vecs, k=self.summarization_perc)

        model = create_empty_model(run_name)
        ranker = fit_ranker(ranker=model, vectors=vecs, metadata=df)
        save_ranker(ranker, name=self.vector_type)

    def _prepare_dataset(self):
        try:
            from sadedegel.dataset import load_raw_corpus, load_annotated_corpus
        except Exception as e:
            raise ValueError("Cannot import raw and annotated corpi.")

        annot = load_annotated_corpus()
        annot_, annot = tee(annot)

        embs = []
        metadata = []
        Doc = DocBuilder()
        for doc_id, doc in track(enumerate(annot), description="Processing documents", total=len(list(annot_))):

            relevance_scores = doc["relevance"]
            d = Doc.from_sentences(doc["sentences"])
            sents = list(d)

            for sent_id, sent in enumerate(sents):
                instance = dict()
                instance["doc_id"] = doc_id
                instance["sent_id"] = sent_id
                instance["relevance"] = relevance_scores[sent_id]

                metadata.append(instance)

            if self.vector_type not in ["tfidf", "bm25"]:
                doc_sent_embeddings = self._get_pretrained_embeddings(sents)
            else:
                raise NotImplementedError("BoW interface for SupervisedSentenceRanker is not yet implemented.")

            embs.append(doc_sent_embeddings)

        df = pd.DataFrame().from_records(metadata)
        vecs = np.vstack(embs)

        return df, vecs
