from ._base import BaseClassifier
from pathlib import Path
import os
import joblib
from ..bblock import Doc
import numpy as np
from sadedegel.summarize import Rouge1Summarizer


class NewsClassifier(BaseClassifier):

    tags = BaseClassifier.tags + ['news', 'tscorpus']

    def __init__(self, base_model="MultinomialNB", embedding_type="bert"):
        self.base_model = base_model
        self.embedding_type = embedding_type

    def _load_model(self):
        modelpath = Path(os.path.dirname(__file__)) / "models" / f"{self.base_model}_{self.embedding_type}.joblib"
        if os.path.exists(modelpath):
            return joblib.load(modelpath)
        else:
            raise FileNotFoundError("Cannot find a trained model! "
                                    "Train and dump a model using SadedeGel classification API.")

    def _get_embeddings(self, document: Doc):
        if self.embedding_type == "bert":
            weights = Rouge1Summarizer().predict(document).reshape(-1, 1)
            emb = np.sum(weights * document.bert_embeddings, axis=0)
        else:
            raise NotImplementedError("Other embeddings are not implemented for SadedeGel classification API")
        return emb

    def _predict(self, embeddings):
        model = self._load_model()
        return model.predict(embeddings)
