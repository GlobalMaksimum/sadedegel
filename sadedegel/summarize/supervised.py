from typing import List
import numpy as np

from .spv import create_model, load_model

from ._base import ExtractiveSummarizer
from sadedegel.bblock import Sentences
from ..config import tokenizer_context


class SupervisedSummarizer(ExtractiveSummarizer):
    tags = ExtractiveSummarizer.tags + ['ml', 'supervised']

    def __init__(self, model_type='rf', normalize=True):
        super(SupervisedSummarizer, self).__init__(normalize)
        self.model_type = model_type

    def _predict(self, sentences: List[Sentences]) -> np.ndarray:
        with tokenizer_context('bert', warning=True):
            if len(sentences) == 0:
                raise ValueError(f"Ensure that document contains a few sentences for summarization")

            model = load_model(self.model_type)

            doc = sentences[0].document

            return model.predict(doc.bert_embeddings)
