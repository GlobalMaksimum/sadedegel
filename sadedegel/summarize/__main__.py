from collections import defaultdict
from math import ceil
from typing import List

from tqdm import tqdm  # type: ignore

import click
from tabulate import tabulate
import numpy as np  # type: ignore
from sklearn.metrics import ndcg_score  # type: ignore

from ..dataset import load_annotated_corpus
from ..summarize import RandomSummarizer, PositionSummarizer, Rouge1Summarizer, KMeansSummarizer, AutoKMeansSummarizer, \
    DecomposedKMeansSummarizer, LengthSummarizer
from ..bblock import Sentences, Doc


def to_sentence_list(sents: List[str]) -> List[Sentences]:
    l: List[Sentences] = []

    for i, sent in enumerate(sents):
        l.append(Sentences(i, sent, l))

    return l


@click.group(help="SadedeGel summarizer commandline")
def cli():
    pass


@cli.command()
@click.option("-f", "--table-format", default="github")
def evaluate(table_format):
    """Evaluate all summarizers in sadedeGel"""
    anno = load_annotated_corpus(return_iter=False)

    scores = defaultdict(list)
    for name, summarizer in tqdm(
            [('Random Summarizer', RandomSummarizer()), ('FirstK Summarizer', PositionSummarizer()),
             ('LastK Summarizer', PositionSummarizer('last')), ('Rouge1 Summarizer (f1)', Rouge1Summarizer()),
             ('Rouge1 Summarizer (precision)', Rouge1Summarizer('precision')),
             ('Rouge1 Summarizer (recall)', Rouge1Summarizer('recall')),
             ('Length Summarizer (char)', LengthSummarizer('token')),
             ('Length Summarizer (token)', LengthSummarizer('char')),
             ('KMeans Summarizer', KMeansSummarizer()),
             ('AutoKMeans Summarizer', AutoKMeansSummarizer()),
             ('DecomposedKMeans Summarizer', DecomposedKMeansSummarizer())], unit=" method",
            desc="Evaluate all summarization methods"):
        for doc in tqdm(anno, unit=" doc", desc=f"Calculate n-dcg score for {name}"):
            y_true = [doc['relevance']]

            d = Doc.from_sentences(doc['sentences'])

            y_pred = [summarizer.predict(d.sents)]

            score_10 = ndcg_score(y_true, y_pred, k=ceil(len(doc['sentences']) * 0.1))
            score_50 = ndcg_score(y_true, y_pred, k=ceil(len(doc['sentences']) * 0.5))
            score_80 = ndcg_score(y_true, y_pred, k=ceil(len(doc['sentences']) * 0.8))

            scores[name].append((score_10, score_50, score_80))

    table = [[algo, np.array([s[0] for s in scores]).mean(), np.array([s[1] for s in scores]).mean(),
              np.array([s[2] for s in scores]).mean()] for
             algo, scores in scores.items()]

    # TODO: Sample weight of instances.
    print(
        tabulate(table, headers=['Method', 'ndcg(k=0.1)', 'ndcg(k=0.5)', 'ndcg(k=0.8)'], tablefmt=table_format,
                 floatfmt=".4f"))


if __name__ == '__main__':
    cli()
