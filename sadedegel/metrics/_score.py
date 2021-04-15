from collections import Counter
from typing import List
import warnings

_AVAILABLE_METRICS = ["f1", "recall", "precision"]


def _get_overlap_count(y_ref: List, y_cand: List) -> int:
    counter_ref = Counter(y_ref)
    counter_cand = Counter(y_cand)

    return sum((counter_ref & counter_cand).values())


def _get_recall(y_ref: list, y_cand: list) -> float:
    if len(y_ref) == 0:
        warnings.warn("y_ref is empty causing division by zero", UserWarning)
        return 0
    else:
        overlap_count = _get_overlap_count(y_ref, y_cand)
        return overlap_count / len(y_ref)


def _get_precision(y_ref: list, y_cand: list) -> float:
    if len(y_cand) == 0:
        warnings.warn("y_ref is empty causing division by zero", UserWarning)
        return 0
    else:
        overlap_count = _get_overlap_count(y_ref, y_cand)
        return overlap_count / len(y_cand)


def _get_f1(y_ref: list, y_cand: list) -> float:
    recall = _get_recall(y_ref, y_cand)
    precision = _get_precision(y_ref, y_cand)

    if recall == 0 and precision == 0:
        warnings.warn("Both precision & recall is 0 causing division by zero in evaluation f1-score", UserWarning)
        f1 = 0.
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    return f1


def rouge1_score(y_ref: List, y_cand: List, metric: str = "f1") -> float:
    if metric.lower() not in _AVAILABLE_METRICS:
        raise ValueError(f"metrics ({metric}) should be one of {_AVAILABLE_METRICS}")

    if not (type(y_ref) == list and type(y_cand) == list):
        raise ValueError("Both inputs (y_ref & y_cand) should be of list type.")

    if metric == "recall":
        return _get_recall(y_ref, y_cand)
    elif metric == "precision":
        return _get_precision(y_ref, y_cand)
    else:
        return _get_f1(y_ref, y_cand)
