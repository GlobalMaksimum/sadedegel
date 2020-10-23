"""
    Wrapper around Facebook's FastText library for word vectorization.
"""

import numpy as np
import fasttext

class FastText:
    def __init__(self, model_path: str = None, use_lower: bool = True):
        self._model = None
        if model_path is not None:
            self._model = fasttext.load_model(model_path)

        # enable if model was trained on lowercase tokens only
        self.use_lower = use_lower


    def __getitem__(self, token: str):
        assert self._model is not None, "Train/load a FastText model before trying to access its vectors!"

        return self._model[token]


    @property
    def words(self):
        return self._model.words

    @staticmethod
    def similarity(v1: np.ndarray, v2: np.ndarray):
        cos_sim = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

        return cos_sim

    @staticmethod
    def _str_dissimilarity(s1: str, s2: str):
        # pseudo-hamming distance
        # truncate longest string to shortest
        if len(s1) > len(s2):
            s1 = s1[:len(s2)]
        else:
            s2 = s2[:len(s1)]

        score = sum((s1[i] == s2[i] for i in range(len(s1))))

        return 1 - score/len(s1)

    def most_similar_to_token(self, token: str, k: int = 5, get_dissimilar_words: bool = False):
        assert k > 0, "Need to get atleast 1 element"

        # ignore top vector as it is the token's vector itself
        if not get_dissimilar_words:
            return self.most_similar_to_vector(self[token], k=k+1)[1:]
        else:
            similarities = []

            token_v = self[token]
            for w in self.words:
                if token != w:
                    similarities.append( (w, self.similarity(token_v, self[w]), self._str_dissimilarity(token, w)) )

            if k > 1:
                return sorted(similarities, key=lambda x: (x[2], x[1]), reverse=True)[:k]
            else:
                return [max(similarities, key=lambda x: (x[2], x[1]))]

    def most_similar_to_vector(self, v: np.ndarray, k: int = 5):
        ## TODO: Use maxheap for faster sorting.
        ## Python's implementation does not allow for custom keys, so need to roll our own


        similarities = []

        for w in self.words:
            similarities.append((w, self.similarity(v, self[w])))

        if k > 1:
            return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
        else:
            return [max(similarities, key=lambda x: x[1])]
