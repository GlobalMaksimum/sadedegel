from typing import List

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ._base import ExtractiveSummarizer
from .util.power_method import degree_centrality_scores
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
        if len(sentences) == 0:
            raise ValueError(f"Ensure that document contains a few sentences for summarization")

        if sentences[0].tokenizer.__name__ != "BertTokenizer":
            with tokenizer_context('bert', warning=True) as Doc2:
                doc = Doc2(sentences[0].document.raw)
        else:
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


class LexRankSummarizer(ExtractiveSummarizer):
    """
        Unsupervised summarizer which just like Rouge1 summarizer
        uses sentence's similarity to other sentences.
    """

    tags = ExtractiveSummarizer.tags + ['ml', 'rank', 'graph']

    def __init__(self, tf_method, idf_method, threshold=.03, fast_power_method=True, normalize=True,
                 **kwargs):
        super().__init__(normalize)

        self.tf_method = tf_method
        self.idf_method = idf_method

        if not (threshold is None or isinstance(threshold, float) and 0 <= threshold < 1):
            raise ValueError("'threshold' should be a floating-point number from the interval [0, 1) or None")

        self.threshold = threshold
        self.fast_power_method = fast_power_method

    def _predict(self, sentences: List[Sentences]) -> np.ndarray:
        similarity_matrix = np.zeros((len(sentences), len(sentences)))

        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i == j:
                    continue

                s1 = sentences[i]
                s2 = sentences[j]

                similarity_matrix[i][j] = \
                    cosine_similarity(s1.get_tfidf(self.tf_method, self.idf_method, drop_stopwords=True, lowercase=True,
                                                   drop_suffix=True,
                                                   drop_punct=True).reshape(1, -1),
                                      s2.get_tfidf(self.tf_method, self.idf_method, drop_stopwords=True, lowercase=True,
                                                   drop_suffix=True,
                                                   drop_punct=True).reshape(1,
                                                                            -1))[
                        0, 0]

        scores = degree_centrality_scores(
            similarity_matrix,
            threshold=self.threshold,
            increase_power=self.fast_power_method,
        )

        return scores
