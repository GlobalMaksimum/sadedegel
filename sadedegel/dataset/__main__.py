import click
import os.path
from os.path import join as pjoin
import glob
from loguru import logger
import json
from tqdm import tqdm


@click.group(help="Dataset commandline")
def cli():
    pass


def get_tokenizer(tokenizer: str):
    if tokenizer == 'nltk-punct':
        from ..tokenize import NLTKPunctTokenizer

        return NLTKPunctTokenizer()
    else:
        from ..tokenize import RegexpSentenceTokenizer

        return RegexpSentenceTokenizer()


@cli.command(help="Generate sentences from raw news text docs.")
@click.option('--dataset-dir', help="Dataset directory")
@click.option('--tokenizer', type=click.Choice(['nltk-punct', 're']), default='re',
              help="Sentences tokenizer.")
@click.option('--force', is_flag=True, default=False, help="Overwrite existing json files.")
def sentences(dataset_dir: str, tokenizer: str, force: bool):
    dataset = os.path.dirname(__file__) if dataset_dir is None else dataset_dir

    raws = glob.glob(pjoin(dataset, 'raw', '*.txt'))
    sents_dir = pjoin(dataset, 'sents')

    os.makedirs(sents_dir, exist_ok=True)

    toker = get_tokenizer(tokenizer)
    n = 0

    logger.info("|raw documents|: {}".format(len(raws)))
    logger.info("Sentence tokenizer: {}".format(tokenizer))

    for file in tqdm(raws):
        basename = os.path.basename(file)
        name, _ = os.path.splitext(basename)

        sentences_json_file = pjoin(sents_dir, "{}.json".format(name))

        if not os.path.exists(sentences_json_file) or force:
            with open(sentences_json_file, 'w') as wp, open(file) as fp:
                json.dump(dict(sentences=toker(fp.read())), wp)

            n += 1

    logger.info("Total number of files dumped is {}".format(n))


if __name__ == '__main__':
    cli()
