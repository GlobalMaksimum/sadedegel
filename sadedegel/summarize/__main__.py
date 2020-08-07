from collections import defaultdict
from math import ceil
from typing import List
import click
from tabulate import tabulate
import numpy as np
from sklearn.metrics import ndcg_score
from ..dataset import load_annotated_corpus
from ..summarize import RandomSummarizer, PositionSummarizer, Rouge1Summarizer, KMeansSummarizer, AutoKMeansSummarizer, \
    DecomposedKMeansSummarizer, LengthSummarizer
from sadedegel.bblock import Sentences, Doc


def to_sentence_list(sents: List[str]) -> List[Sentences]:
    l = []

    for i, sent in enumerate(sents):
        l.append(Sentences(i, sent, l))

    return l


@click.group()
def cli():
    pass


@cli.command()
@click.option("-f", "--table-format", default="github")
def evaluate(table_format):
    """Evaluate all summarizers in sadedeGel"""
    anno = load_annotated_corpus()

    scores = defaultdict(list)
    for name, summarizer in [('Random', RandomSummarizer()), ('FirstK', PositionSummarizer()),
                             ('LastK', PositionSummarizer('last')), ('Rouge1 (f1)', Rouge1Summarizer()),
                             ('Rouge1 (precision)', Rouge1Summarizer('precision')),
                             ('Rouge1 (recall)', Rouge1Summarizer('recall')),
                             ('Length (char)', LengthSummarizer('token')),
                             ('Length (token)', LengthSummarizer('char')),
                             ('KMeans', KMeansSummarizer()),
                             ('AutoKMeansSummarizer', AutoKMeansSummarizer()),
                             ('DecomposedKMeansSummarizer', DecomposedKMeansSummarizer())]:
        for doc in anno:
            y_true = [doc['relevance']]

            sents_list = to_sentence_list(doc['sentences'])

            if name in ("KMeans", 'AutoKMeansSummarizer', 'DecomposedKMeansSummarizer'):
                y_pred = [summarizer.predict(Doc(None, doc['sentences']))]
            else:
                y_pred = [summarizer.predict(sents_list)]

            score_10 = ndcg_score(y_true, y_pred, k=ceil(len(doc['sentences']) * 0.1))
            score_50 = ndcg_score(y_true, y_pred, k=ceil(len(doc['sentences']) * 0.5))
            score_80 = ndcg_score(y_true, y_pred, k=ceil(len(doc['sentences']) * 0.8))

            scores[name].append((score_10, score_50, score_80))

    table = [[algo, np.array([s[0] for s in scores]).mean(), np.array([s[1] for s in scores]).mean(),
              np.array([s[2] for s in scores]).mean()] for
             algo, scores in scores.items()]

    # TODO: Sample weigth of instances.
    print(
        tabulate(table, headers=['Method', 'ndcg(k=0.1)', 'ndcg(k=0.5)', 'ndcg(k=0.8)'], tablefmt=table_format,
                 floatfmt=".4f"))


if __name__ == '__main__':
    cli()
