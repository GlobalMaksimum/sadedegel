import click
from gensim.models import Word2Vec
from multiprocessing import cpu_count
from pathlib import Path
from os.path import dirname
from tqdm import tqdm
from .word2vec_utils import GCorpus


@click.group(help="Gensim Word2Vec Commandline")
def cli():
    pass


@cli.command(help="Train a gensim based word2vec model.")
@click.option('--model-name', '-m', default='gensim_model')
@click.option('--corpus', '-c', type=click.Choice(['standard', 'extended', 'tokenization']), default='standard')
@click.option('--tokenizer', '-t', type=click.Choice(['simple', 'bert']), default='simple')
@click.option('--num-epochs', '-e', help='Training epochs', default=10)
@click.option('--skip-gram', '-s', help='Skip Gram or CBOW. Defaults to True for Skip Gram', default=True)
@click.option('--retrain-from', '-r', default=None)
def train_word2vec(model_name, corpus, tokenizer, num_epochs, skip_gram, retrain_from):

    if not retrain_from:
        sentences = GCorpus(sadedegel_corpus=corpus, tokenizer=tokenizer)
        model = Word2Vec(size=100,
                         workers=cpu_count(),
                         min_count=3,
                         sg=skip_gram,
                         seed=42)
        click.secho(click.style('Building Vocab...', fg='yellow'))
        model.build_vocab(sentences)

        click.secho(click.style('Training model...', fg='yellow'))
        for e in tqdm(range(num_epochs)):
            sentences = GCorpus(sadedegel_corpus=corpus, tokenizer=tokenizer)
            model.train(sentences,
                        epochs=1,
                        total_examples=model.corpus_count,
                        report_delay=1)

        model_name += '.model'
        modelpath = (Path(dirname(__file__)) / 'model' / tokenizer / model_name).absolute()
        click.secho(f"Saving model to "+click.style(f"{modelpath}", fg='blue'), color='white')
        model.save(str(modelpath))

    else:
        model_name += '.model'
        modelpath = (Path(dirname(__file__)) / 'model' / tokenizer / model_name).absolute()
        click.secho(f"Loading model from " + click.style(f"{modelpath}", fg='blue'), color='white')
        model = Word2Vec.load(str(modelpath))

        sentences = GCorpus(sadedegel_corpus=corpus, tokenizer=tokenizer)

        model.build_vocab(sentences, update=True)

        click.secho(click.style('Training model...', fg='yellow'))
        for e in tqdm(range(num_epochs)):
            sentences = GCorpus(sadedegel_corpus=corpus, tokenizer=tokenizer)
            model.train(sentences,
                        epochs=1,
                        total_examples=model.corpus_count,
                        report_delay=1)


if __name__ == '__main__':
    cli()
