from collections import defaultdict
from typing import List
from math import log
import numpy as np

from ..tokenize import Doc, Sentences
from ._base import ExtractiveSummarizer


class TFIDFSummarizer(ExtractiveSummarizer):
    def __init__(self, all_docs: List[str], normalize=False):
        super().__init__(normalize)
        all_words_dict = defaultdict(lambda: 0)
        doc_count = 0
        for doc_str in all_docs:
            doc = Doc(doc_str)
            unique_words = set()
            for sent in doc.sents:
                for token in sent.tokens:
                    unique_words.add(token)
            for token in unique_words:
                all_words_dict[token] += 1
            doc_count += 1
        self.all_words_dict = all_words_dict
        self.doc_count = doc_count

    def _predict(self, sents: List[Sentences]):
        doc_word_dict = defaultdict(lambda: 0)
        doc_word_count = 0
        for sent in sents:
            for token in sent.tokens:
                doc_word_dict[token] += 1
                doc_word_count += 1
        scores = [None] * len(sents)
        i = 0
        for sent in sents:
            scores[i] = []
            for token in sent.tokens:
                scores[i].append((doc_word_dict[token] / doc_word_count) * log(
                                 (self.doc_count / (1 + self.all_words_dict[token]))))
            scores[i] = sum(scores[i])
            i += 1
        return np.array(scores)
