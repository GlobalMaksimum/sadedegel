from .util import grid_search
from tabulate import tabulate

from ...dataset import load_annotated_corpus
from ...config import config_context
from .parameter_space import bm25_parameter_space, tfidf_parameter_space, lexrank_parameter_space, \
    textrank_parameter_space
from ..bm25 import BM25Summarizer
from ..tf_idf import TFIDFSummarizer
from ..baseline import BandSummarizer, RandomSummarizer, PositionSummarizer, LengthSummarizer
from ..rouge import Rouge1Summarizer
from ..rank import TextRank, LexRankSummarizer

import click

summarizer_list = [(BM25Summarizer, bm25_parameter_space, 50),
                   (TFIDFSummarizer, tfidf_parameter_space, 50),
                   (BandSummarizer, lambda x: [dict(k=i) for i in range(2, 6)], 5),
                   (RandomSummarizer, lambda x: [{"normalize": True}], 1),
                   (PositionSummarizer, lambda x: [dict(mode="first"), dict(mode="last")], 2),
                   (
                       Rouge1Summarizer, lambda x: [dict(metric="precision"), dict(metric="recall"), dict(metric="f1")],
                       3),
                   (LengthSummarizer, lambda x: [dict(mode="token"), dict(mode="char")], 2),
                   (LexRankSummarizer, lexrank_parameter_space, 10),
                   (TextRank, textrank_parameter_space, 10)]


@click.group(help="SadedeGel summarizer commandline")
def cli():
    pass


@cli.command()
@click.option("-f", "--table-format", default="github")
@click.option("-t", "--tag", default=["extractive"], multiple=True)
@click.option("-d", "--debug", default=False)
@click.option("--tokenizer", default="bert")
def evaluate(table_format, tag, debug, tokenizer):
    """Evaluate various extractive summarizer using random hyper parameter sampling"""

    anno = load_annotated_corpus(False)
    relevance = [[doc['relevance']] for doc in anno]

    with config_context(tokenizer=tokenizer) as DocBuilder:
        docs = [DocBuilder.from_sentences(doc['sentences']) for doc in anno]

    table = []

    for Summarizer, parameter_space, n_trial in filter(lambda x: any((t in x[0].tags) for t in tag), summarizer_list):
        if 'bert' in Summarizer.tags and tokenizer != 'bert':
            continue

        scores = grid_search(relevance, docs, Summarizer, parameter_space(n_trial))

        table += [[method[0], method[1], scores[0], scores[1], scores[2]] for
                  method, scores in scores.items()]

    print(
        tabulate(sorted(table, key=lambda x: x[2], reverse=True)[:5]
                 , headers=['Method', "Parameter", 'ndcg(k=0.1)', 'ndcg(k=0.5)', 'ndcg(k=0.8)'],
                 tablefmt="github",
                 floatfmt=".4f"))

    print(
        tabulate(sorted(table, key=lambda x: x[3], reverse=True)[:5]
                 , headers=['Method', "Parameter", 'ndcg(k=0.1)', 'ndcg(k=0.5)', 'ndcg(k=0.8)'],
                 tablefmt="github",
                 floatfmt=".4f"))

    print(
        tabulate(sorted(table, key=lambda x: x[4], reverse=True)[:5]
                 , headers=['Method', "Parameter", 'ndcg(k=0.1)', 'ndcg(k=0.5)', 'ndcg(k=0.8)'],
                 tablefmt="github",
                 floatfmt=".4f"))


if __name__ == '__main__':
    cli()
