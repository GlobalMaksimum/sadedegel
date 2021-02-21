import json
from math import ceil
from typing import List, Tuple

import numpy as np
from rich.live import Live
from rich.table import Table
from rich.console import Console
from sklearn.metrics import ndcg_score  # type: ignore

console = Console()


def topk(scores: dict, k: int = 3) -> List[Tuple[str, Tuple[float, float, float]]]:
    return sorted(scores.items(), key=lambda kv: kv[1][1], reverse=True)[:k]


def topk_table(scores: dict, i: int, total: int, k: int = 3):
    """Make a new table."""
    table = Table(title=f"{i}/{total} completed...")
    table.add_column("algorithm")
    table.add_column("parameters")
    table.add_column("ndcg (k=0.1)")
    table.add_column("ndcg (k=0.5)")
    table.add_column("ndcg (k=0.8)")

    for k, v in topk(scores, k):
        table.add_row(k[0], k[1], str(v[0]), str(v[1]), str(v[2]))

    return table


def grid_search(relevance, docs, summarize_class, parameter_space):
    scores = {}

    with Live(console=console, screen=True, auto_refresh=False) as live:
        for i, param in enumerate(parameter_space, 1):
            summarizer = summarize_class(**param)

            score_10, score_50, score_80 = [], [], []

            for y_true, d in zip(relevance, docs):
                y_pred = [summarizer.predict(d)]

                score_10.append(ndcg_score(y_true, y_pred, k=ceil(len(d) * 0.1)))
                score_50.append(ndcg_score(y_true, y_pred, k=ceil(len(d) * 0.5)))
                score_80.append(ndcg_score(y_true, y_pred, k=ceil(len(d) * 0.8)))

            scores[(summarizer.__class__.__name__, json.dumps(param))] = np.array(score_10).mean(), np.array(
                score_50).mean(), np.array(score_80).mean()

            live.update(topk_table(scores, i, len(parameter_space)), refresh=True)

    return scores
