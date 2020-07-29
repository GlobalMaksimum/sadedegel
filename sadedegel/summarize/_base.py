from typing import List, Union
import numpy as np
from ..tokenize.helper import Doc, Sentences

class ExtractiveSummarizer:
    def predict(self, sents: List[Sentences]) -> np.ndarray:
        raise NotImplementedError("ExtractiveSummarizer .predict() is an abstract method!")

    def __call__(self, sents: Union[Doc, List[Sentences]], k: int) -> List[Sentences]:
        if type(sents) == Doc:
            sents = sents.sents # Get the list of Sentences from Doc

        scores = self.predict(sents)
        topk_inds = np.argpartition(scores, k)[-k:] # returns indices of k top sentences
        topk_inds.sort() # fix the order of sentences

        summ = [sents[i] for i in topk_inds]
        return summ
