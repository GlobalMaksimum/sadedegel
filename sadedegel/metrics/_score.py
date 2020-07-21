from collections import Counter
from typing import List

_AVAILABLE_METRICS = ["f1", "recall", "precision"]


def _get_overlap_count(y_ref: List, y_cand: List) -> int:
    counter_ref = Counter(y_ref)
    counter_cand = Counter(y_cand)

    return sum((counter_ref & counter_cand).values())


def _get_recall(y_ref: list, y_cand: list) -> float:
    overlap_count = _get_overlap_count(y_ref, y_cand)
    return overlap_count / len(y_ref)


def _get_precision(y_ref: list, y_cand: list) -> float:
    overlap_count = _get_overlap_count(y_ref, y_cand)
    return overlap_count / len(y_cand)


def _get_f1(y_ref: list, y_cand: list) -> float:
    recall = _get_recall(y_ref, y_cand)
    precision = _get_precision(y_ref, y_cand)

    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def rouge1_score(y_ref: List, y_cand: List, metric: str = "f1"):
    if metric.lower() not in _AVAILABLE_METRICS:
        raise ValueError(f"metrics ({metric}) should be one of {_AVAILABLE_METRICS}")

    if type(y_cand) == list and type(y_cand) == list:

        if metric == "recall":
            return _get_recall(y_ref, y_cand)
        elif metric == "precision":
            return _get_precision(y_ref, y_cand)
        else:
            return _get_f1(y_ref, y_cand)
