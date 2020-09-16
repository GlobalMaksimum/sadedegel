from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from ._base import ExtractiveSummarizer
from ..bblock import Sentences
from ..config import tokenizer_context


class TextRank(ExtractiveSummarizer):
    """Assign a higher importance score to longer sentences.

    mode : {'token', 'char'}, default='token'
        Whether the distance should be defined by total number of characters or
        total number of tokens in a sentence.

    normalize : bool, optional (default=True)
        If ``False``, return a raw score vector.
        Otherwise, return a score vector that adds up to 1.
    """

    tags = ExtractiveSummarizer.tags + ['ml', 'rank', 'graph']

    def __init__(self, input_type="bert", alpha=0.5, normalize=True):
        super().__init__(normalize)

        if input_type not in ['bert']:
            raise ValueError(f"mode should be one of 'bert'")

        self.input_type = input_type
        self.alpha = alpha

    def _predict(self, sentences: List[Sentences]) -> np.ndarray:
        with tokenizer_context('bert', warning=True):
            if len(sentences) == 0:
                raise ValueError(f"Ensure that document contains a few sentences for summarization")

            doc = sentences[0].document

            vectors = doc.bert_embeddings

            similarity_matrix = np.zeros((len(vectors), len(vectors)))

            for i in range(len(vectors)):
                for j in range(len(vectors)):
                    if i == j:
                        continue

                    similarity_matrix[i][j] = cosine_similarity(vectors[i].reshape(1, -1), vectors[j].reshape(1, -1))[
                        0, 0]

            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph, alpha=self.alpha)

            return np.array([scores[i] for i in range(len(scores))])
