from collections import defaultdict
from math import ceil
from typing import List
from loguru import logger

import click
from tabulate import tabulate
import warnings
import numpy as np  # type: ignore
from sklearn.metrics import ndcg_score  # type: ignore

from sadedegel.dataset import load_annotated_corpus
from sadedegel.summarize import RandomSummarizer, PositionSummarizer, Rouge1Summarizer, KMeansSummarizer, \
    AutoKMeansSummarizer, \
    DecomposedKMeansSummarizer, LengthSummarizer, TextRank, TFIDFSummarizer
from sadedegel import Sentences, Doc
from sadedegel import tokenizer_context

logger.disable("sadedegel")

SUMMARIZERS = [('Random Summarizer', RandomSummarizer()), ('FirstK Summarizer', PositionSummarizer()),
               ('LastK Summarizer', PositionSummarizer('last')), ('Rouge1 Summarizer (f1)', Rouge1Summarizer()),
               ('Rouge1 Summarizer (precision)', Rouge1Summarizer('precision')),
               ('Rouge1 Summarizer (recall)', Rouge1Summarizer('recall')),
               ('Length Summarizer (char)', LengthSummarizer('token')),
               ('Length Summarizer (token)', LengthSummarizer('char')),
               ('KMeans Summarizer', KMeansSummarizer()),
               ('AutoKMeans Summarizer', AutoKMeansSummarizer()),
               ('DecomposedKMeans Summarizer', DecomposedKMeansSummarizer()),
               ("TextRank(0.05) Summarizer (BERT)", TextRank(alpha=0.05)),
               ("TextRank(0.15) Summarizer (BERT)", TextRank(alpha=0.15)),
               ("TextRank(0.30) Summarizer (BERT)", TextRank(alpha=0.30)),
               ("TextRank(0.5) Summarizer (BERT)", TextRank(alpha=0.5)),
               ("TextRank(0.6) Summarizer (BERT)", TextRank(alpha=0.6)),
               ("TextRank(0.7) Summarizer (BERT)", TextRank(alpha=0.7)),
               ("TextRank(0.85) Summarizer (BERT)", TextRank(alpha=0.85)),
               ("TextRank(0.9) Summarizer (BERT)", TextRank(alpha=0.9)),
               ("TextRank(0.95) Summarizer (BERT)", TextRank(alpha=0.95)),
               ("TFIDF Summarizer", TFIDFSummarizer())]


def to_sentence_list(sents: List[str]) -> List[Sentences]:
    l: List[Sentences] = []

    for i, sent in enumerate(sents):
        l.append(Sentences(i, sent, l))

    return l


@click.group(help="SadedeGel summarizer commandline")
def cli():
    pass


def dot_progress(i, length):
    n = ceil(length * 0.05)

    if i == length - 1:
        click.echo(".", nl=True, color="yellow")
    elif i % n == 0 and i != 0:
        click.echo(".", nl=False, color="yellow")


@cli.command()
@click.option("-f", "--table-format", default="github")
@click.option("-t", "--tag", default=["extractive"], multiple=True)
@click.option("-d", "--debug", default=False)
def evaluate(table_format, tag, debug):
    """Evaluate all summarizers in sadedeGel"""
    if not debug:
        warnings.filterwarnings("ignore")

    anno = load_annotated_corpus(False)
    relevance = [[doc['relevance']] for doc in anno]

    summarizers = [summ for summ in SUMMARIZERS if any(_tag in summ[1] for _tag in tag)]

    scores = defaultdict(list)

    for word_tokenizer in ['simple', 'bert']:
        click.echo("Word Tokenizer: " + click.style(f"{word_tokenizer}", fg="blue"))
        docs = [Doc.from_sentences(doc['sentences']) for doc in anno]  # Reset document because of memoization
        with tokenizer_context(word_tokenizer):
            for name, summarizer in summarizers:
                click.echo(click.style(f"    {name} ", fg="magenta"), nl=False)
                # skip simple tokenizer for clustering models
                if ("cluster" in summarizer or "rank" in summarizer or name == "TFIDF Summarizer") and \
                        word_tokenizer == "simple":
                    click.echo(click.style("SKIP", fg="yellow"))
                    continue

                for i, (y_true, d) in enumerate(zip(relevance, docs)):
                    dot_progress(i, len(relevance))

                    y_pred = [summarizer.predict(d.sents)]

                    score_10 = ndcg_score(y_true, y_pred, k=ceil(len(d) * 0.1))
                    score_50 = ndcg_score(y_true, y_pred, k=ceil(len(d) * 0.5))
                    score_80 = ndcg_score(y_true, y_pred, k=ceil(len(d) * 0.8))

                    scores[f"{name} - {word_tokenizer}"].append((score_10, score_50, score_80))

    table = [[algo, np.array([s[0] for s in scores]).mean(), np.array([s[1] for s in scores]).mean(),
              np.array([s[2] for s in scores]).mean()] for
             algo, scores in scores.items()]

    # TODO: Sample weight of instances.
    print(
        tabulate(table, headers=['Method & Tokenizer', 'ndcg(k=0.1)', 'ndcg(k=0.5)', 'ndcg(k=0.8)'],
                 tablefmt=table_format,
                 floatfmt=".4f"))

    if debug:
        click.echo(np.array(table).shape)


if __name__ == '__main__':
    cli()
