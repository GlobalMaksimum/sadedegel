import sys
from collections import Counter
from collections import defaultdict
from itertools import islice
from itertools import tee
from multiprocessing import cpu_count
from typing import Tuple

import click
import numpy as np
from rich.console import Console
from rich.progress import track
from rich.table import Table
from tqdm import tqdm

from ..vocabulary import VocabularyCounter
from ..word_tokenizer_helper import word_tokenize, ICUTokenizerHelper
from ...config import tokenizer_context
from ...dataset.extended import load_extended_sents_corpus
from ...dataset.tscorpus import load_tokenization_tokenized, load_tokenization_raw, fail_with_corpus_validation, \
    CORPUS_SIZE


def get_w2v_model(size: int = 100, min_count=3, skip_gram=True, workers: int = cpu_count(), seed=42):
    try:
        from gensim.models import Word2Vec
    except ImportError:
        print(
            ("Error in importing gensim module. "
             "Ensure that you run 'pip install sadedegel[gensim]' to enable word2vec training"),
            file=sys.stderr)
        sys.exit(1)

    return Word2Vec(size=size, workers=workers, sg=skip_gram, min_count=min_count, iter=iter, seed=seed)


console = Console()


def tok_eval(tokenizer, limit=None) -> Tuple[str, float, float]:
    """Evaluate tokenizers on a downsized version of tokenization dataset.

    :type tokenizer: str
    :param limit: Top-K documents for evaluation
    :type limit: int

    :return: IoU score between list of true tokens and list of tokenized tokens.
    """
    if tokenizer == "bert":
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError:
            console.print(
                ("Error in importing transformers module. "
                 "Ensure that you run 'pip install sadedegel[bert]' to use BERT features."),
                file=sys.stderr)
            sys.exit(1)

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

    return tokenizer, np.mean(macro), micro['intersection'] / micro['union']


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
        table.add_row(row[0], f"{row[1]:.4f}", f"{row[2]:.4f}")

    console.print(table)


def sentence_iter(word_tokenizer, max_doc=None):
    if max_doc is None:
        corpus = load_extended_sents_corpus()
    else:
        corpus = islice(load_extended_sents_corpus(), max_doc)

    with tokenizer_context(word_tokenizer) as Doc:
        for d in corpus:
            doc = Doc.from_sentences(d['sentences'])

            for sent in doc:
                yield [t.lower_ for t in sent if not t.is_punct]


@cli.command()
@click.option('--max-doc', help="Maximum number of documents in extended corpus", type=int, default=None)
@click.option('--min-df', help="Minimum document frequency of a word to be included in", default=3)
@click.option('--min-freq', help="Minimum term frequency", default=3)
@click.option('--word-tokenizer', "-t", type=click.Choice(['bert', "icu", "simple"], case_sensitive=False),
              help="Word tokenizer to be used in building vocabulary.", default='bert')
@click.option('--w2v/--no-w2v', default=True, help="Train word embeddings")
@click.option('--w2v-num-epoch', type=int, help="Number of epochs in word2vec training", default=10)
@click.option('--w2v-skip-gram', type=int, help='Skip Gram or CBOW. Defaults to True for Skip Gram', default=True)
@click.option('--w2v-size', type=int, help='Dimension of word vectors', default=100)
def build_vocabulary(max_doc, min_df, min_freq, word_tokenizer, w2v, w2v_num_epoch, w2v_skip_gram, w2v_size):
    """Build vocabulary & optionally word embeddings"""

    total = sum(1 for _ in sentence_iter(word_tokenizer, max_doc))

    counter_cs = VocabularyCounter(word_tokenizer, case_sensitive=True, min_tf=min_freq, min_df=min_df)
    counter = VocabularyCounter(word_tokenizer, case_sensitive=False, min_tf=min_freq, min_df=min_df)

    if w2v:
        model = get_w2v_model(w2v_size, min_freq, w2v_skip_gram)
        console.log("Building vocabulary...")
        model.build_vocab(sentence_iter(word_tokenizer, max_doc))

        for _ in track(range(w2v_num_epoch), total=w2v_num_epoch, description="Building word vectors..."):
            model.train(sentences=sentence_iter(word_tokenizer, max_doc),
                        epochs=1,
                        total_examples=total,
                        report_delay=1)

    click.secho(click.style(f"...Frequency calculation over extended dataset", fg="blue"))
    with tokenizer_context(word_tokenizer) as Doc:
        if max_doc is None:
            corpus = load_extended_sents_corpus()
        else:
            corpus = islice(load_extended_sents_corpus(), max_doc)

        corpus, X = tee(corpus, 2)

        for i, d in track(enumerate(corpus),
                          total=sum(1 for _ in X),
                          description="Building vocabulary..."):
            doc = Doc.from_sentences(d['sentences'])

            for sent in doc:
                for word in sent.tokens:
                    counter.inc(word, i, 1)
                    counter_cs.inc(word, i, 1)

    counter = counter.prune()
    counter_cs = counter_cs.prune()

    if w2v:
        counter.to_hdf5(model)
    else:
        counter.to_hdf5()
    counter_cs.to_hdf5()


if __name__ == '__main__':
    cli()
