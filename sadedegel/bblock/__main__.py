from tqdm import tqdm
from itertools import islice
import click
from sadedegel.config import tokenizer_context

from .. import Doc
from ..dataset.extended import load_extended_sents_corpus
from .vocabulary import Vocabulary


@click.command()
@click.option('--max-doc', help="Maximum documentation in extended corpus", type=int, default=-1)
@click.option('--min-df', help="Minimum document frequent of a word to be included in", default=3)
@click.option('--word-tokenizer', type=click.Choice(['bert', 'simple'], case_sensitive=False),
              help="Word tokenizer to be used in building vocabulary.", default='bert')
def build_vocabulary(max_doc, min_df, word_tokenizer):
    """Build vocabulary.
    """
    with tokenizer_context(word_tokenizer):
        if max_doc > 0:
            corpus = islice(load_extended_sents_corpus(), max_doc)
        else:
            corpus = load_extended_sents_corpus()

        vocab = Vocabulary.factory(word_tokenizer)

        click.secho(click.style(f"...Frequency calculation over extended dataset", fg="blue"))

        for i, d in tqdm(enumerate(corpus), unit=" doc"):
            doc = Doc.from_sentences(d['sentences'])

            for sent in doc:
                for word in sent.tokens:
                    vocab.add_word_to_doc(word, i)

        vocab.build(min_df)
        vocab.save()

        click.secho(click.style(f"Total documents {vocab.document_count}", fg="blue"))
        click.secho(click.style(f"Vocabulary size {len(vocab)} (words occured more than {min_df} documents)", fg="blue"))


if __name__ == '__main__':
    build_vocabulary()
