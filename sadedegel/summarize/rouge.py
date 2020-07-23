import numpy as np


class Rouge1Summarizer:
    """Assign a higher importance score based on ROUGE1 score of the sentences within the document.

    metric : {'f1', 'precision','recall'}, default='f1'
        Metric to be used for ROUGE1 computation.

    normalize : bool, optional (default=True)
        If ``False``, return a raw score vector.
        Otherwise, return L2 normalized score vector.
    """

    def __init__(self, metric='f1', normalize=True):
        self.normalize = normalize
        if metric not in ['f1', 'precision', 'recall']:
            raise Exception(f"mode should be one of 'first', 'last'")

        self.metric = metric

    def predict(self, sentences):
        scores = [sent.rouge1(self.metic) for sent in sentences]

        if self.normalize:
            return scores / np.linalg.norm(scores)
        else:
            return scores
