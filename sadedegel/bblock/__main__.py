from tqdm import tqdm
from collections import defaultdict
from itertools import islice
import click

from .. import Doc
from ..dataset.extended import load_extended_sents_corpus
from .util import tr_lower
from .vocabulary import Vocabulary, Token


@click.command()
@click.option('--max-doc', help="Maximum documentation in extended corpus", type=int, default=-1)
@click.option('--min-df', help="Mininum document frequenct of a word to be included in", default=3)
@click.option('--word-tokenizer', type=click.Choice(['bert'], case_sensitive=False),
              help="Word tokenizer to be used in building vocabulary.", default='bert')
def build_vocabulary(max_doc, min_df, word_tokenizer):
    """Build vocabulary.
    """
    if max_doc > 0:
        corpus = islice(load_extended_sents_corpus(), max_doc)
    else:
        corpus = load_extended_sents_corpus()

    vocab = defaultdict(set)

    n_documents = 0

    for i, d in tqdm(enumerate(corpus), unit=" doc"):
        doc = Doc.from_sentences(d['sentences'])

        for sent in doc.sents:
            for word in sent.tokens:
                vocab[tr_lower(word)].add(i)

        n_documents += 1

    w_i = 0
    for w in vocab:
        if len(vocab[w]) >= min_df:
            Vocabulary.tokens[w] = Token(w_i, w, len(vocab[w]), n_documents)
            w_i += 1

    Vocabulary.size = w_i
    Vocabulary.save()

    click.secho(click.style(f"Total documents {n_documents}", fg="blue"))
    click.secho(click.style(f"Vocabulary size {w_i} (words occured more than {min_df} documents)", fg="blue"))


if __name__ == '__main__':
    build_vocabulary()
