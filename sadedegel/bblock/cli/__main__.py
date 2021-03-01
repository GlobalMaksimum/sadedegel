import click
import numpy as np
from itertools import islice, tee

from collections import defaultdict

from rich.table import Table
from rich.console import Console
from rich.progress import track

from tqdm import tqdm

from ...dataset.tscorpus import load_tokenization_tokenized, load_tokenization_raw, fail_with_corpus_validation, \
    CORPUS_SIZE
from ...dataset.extended import load_extended_sents_corpus
from ...config import tokenizer_context
from ..vocabulary import Vocabulary

from ..icu import ICUTokenizerHelper
from ..word_tokenizer_helper import word_tokenize

from collections import Counter
from typing import Tuple

console = Console()


def tok_eval(tokenizer, limit=None) -> Tuple[str, str, str]:
    """Evaluate tokenizers on a downsized version of tokenization dataset.

    :type tokenizer: str
    :param limit: Top-K documents for evaluation
    :type limit: int

    :return: IoU score between list of true tokens and list of tokenized tokens.
    """
    if tokenizer == "bert":
        import torch
        from transformers import AutoTokenizer
        toker = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased").tokenize

    elif tokenizer == "simple":
        toker = word_tokenize
    else:
        toker = ICUTokenizerHelper()

    console.log(f"[bold blue]Evaluating {tokenizer} tokenizer...")

    if limit:
        raw = islice(load_tokenization_raw(), limit)
        tok = islice(load_tokenization_tokenized(), limit)
    else:
        raw = load_tokenization_raw()
        tok = load_tokenization_tokenized()

    macro = []
    micro = defaultdict(int)
    wmacro = []
    wmicro = defaultdict(int)
    for i, (raw_doc, tokenized_doc) in tqdm(enumerate(zip(raw, tok)), desc="Evaluating...",
                                            total=limit if limit else CORPUS_SIZE):
        if raw_doc['id'] != tokenized_doc['id']:
            fail_with_corpus_validation()

        true_tokens = Counter(tokenized_doc['tokens'])

        tokens = toker(raw_doc['text'])

        if tokenizer == 'bert':
            fixed_hashtags = []
            for tok in tokens:
                if "##" not in tok:
                    fixed_hashtags.append(tok)
                else:
                    merged = fixed_hashtags[-1] + tok.split("##")[1]
                    fixed_hashtags.pop(-1)
                    fixed_hashtags.append(merged)
            tokens = fixed_hashtags

        pred_tokens = Counter(tokens)

        union = sum((pred_tokens | true_tokens).values())
        intersection = sum((pred_tokens & true_tokens).values())

        wmacro.append(intersection / union)

        wmicro['intersection'] += intersection
        wmicro['union'] += union

        union = len(set(pred_tokens) | set(true_tokens))
        intersection = len(set(pred_tokens) & set(true_tokens))

        macro.append(intersection / union)

        micro['intersection'] += intersection
        micro['union'] += union

    # console.log(f"{np.mean(macro):.4f}", f"{micro['intersection'] / micro['union']:.4f}")
    # console.log(f"Weighted: {np.mean(wmacro):.4f}", f"{wmicro['intersection'] / wmicro['union']:.4f}")

    return tokenizer, f"{np.mean(macro):.4f}", f"{micro['intersection'] / micro['union']:.4f}"


@click.group()
def cli():
    pass


@click.option("--tokenizers", "-t", help="List of tokenizers to evaluate", multiple=True)
@click.option("--limit", "-s", help="Evaluation set size", type=int)
@cli.command()
def tokenizer_evaluate(tokenizers, limit):
    """Tokenizer evaluation"""

    result = []
    for tok in tokenizers:
        result.append(tok_eval(tok, limit))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Tokenizer")
    table.add_column("IoU (macro)")
    table.add_column("IoU (micro)")

    for row in result:
        table.add_row(*row)

    console.print(table)


@cli.command()
@click.option('--max-doc', help="Maximum number of documents in extended corpus", type=int, default=-1)
@click.option('--min-df', help="Minimum document frequency of a word to be included in", default=3)
@click.option('--word-tokenizer', "-t", type=click.Choice(['bert', "icu", "simple"], case_sensitive=False),
              help="Word tokenizer to be used in building vocabulary.", default='bert')
def build_vocabulary(max_doc, min_df, word_tokenizer):
    """Build vocabulary"""

    if max_doc > 0:
        corpus = islice(load_extended_sents_corpus(), max_doc)
    else:
        corpus = load_extended_sents_corpus()

    corpus, replica = tee(corpus, 2)
    total = sum(1 for _ in replica)

    vocab = Vocabulary.factory(word_tokenizer)

    click.secho(click.style(f"...Frequency calculation over extended dataset", fg="blue"))

    with tokenizer_context(word_tokenizer) as Doc:
        for i, d in track(enumerate(corpus), total=max_doc if max_doc > 0 else total,
                          description="Building vocabulary..."):
            doc = Doc.from_sentences(d['sentences'])

            for sent in doc:
                for word in sent.tokens:
                    vocab.add_word_to_doc(word, i)

    vocab.build(min_df)
    vocab.save(word_tokenizer)

    click.secho(click.style(f"Total documents {vocab.document_count}", fg="blue"))
    click.secho(click.style(f"Vocabulary size {len(vocab)} (words occurred more than {min_df} documents)", fg="blue"))


if __name__ == '__main__':
    cli()
