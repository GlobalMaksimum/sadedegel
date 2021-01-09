from rich.progress import track
from itertools import islice, tee
import click

from ..dataset.extended import load_extended_sents_corpus
from ..config import tokenizer_context
from .vocabulary import Vocabulary
from ..about import __version__
import warnings


@click.command()
@click.option('--max-doc', help="Maximum documentation in extended corpus", type=int, default=-1)
@click.option('--min-df', help="Mininum document frequenct of a word to be included in", default=3)
@click.option('--word-tokenizer', type=click.Choice(['bert'], case_sensitive=False),
              help="Word tokenizer to be used in building vocabulary.", default='bert')
def build_vocabulary(max_doc, min_df, word_tokenizer):
    """Build vocabulary"""

    if tuple(map(int, __version__.split('.'))) < (0, 18):
        warnings.warn(
            ("sadedegel.bblock.__main__ is deprecated and will be dropped by release 0.18."
             " Please use sadedegel.bblock.cli tokenizer-evaluate"),
            DeprecationWarning,
            stacklevel=2)
    else:
        raise Exception("Remove sadedegel.bblock.__main__ before release.")

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
    vocab.save()

    click.secho(click.style(f"Total documents {vocab.document_count}", fg="blue"))
    click.secho(click.style(f"Vocabulary size {len(vocab)} (words occurred more than {min_df} documents)", fg="blue"))


if __name__ == '__main__':
    build_vocabulary()
