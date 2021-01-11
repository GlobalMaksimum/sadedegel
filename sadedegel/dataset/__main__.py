import click
import os.path
from os.path import join as pjoin
import glob
from loguru import logger
import json
from rich.progress import track
import sys


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

    for file in track(raws, description="Building sentence corpus..."):
        basename = os.path.basename(file)
        name, _ = os.path.splitext(basename)

        sentences_json_file = pjoin(sents_dir, "{}.json".format(name))

        if not os.path.exists(sentences_json_file) or force:
            with open(sentences_json_file, 'w') as wp, open(file) as fp:
                json.dump(dict(sentences=toker(fp.read())), wp)

            n += 1

    logger.info("Total number of files dumped is {}".format(n))


@cli.command(help="Validate raw & sentences datasets")
@click.option("-v", count=True)
@click.option("--base-path", help="Base data path", default=None)
def validate(v, base_path):
    from sadedegel.dataset import load_raw_corpus, load_sentence_corpus, load_annotated_corpus, file_paths, \
        CorpusTypeEnum

    click.secho("Corpus loading...")
    raw = load_raw_corpus(False, base_path)
    sents = load_sentence_corpus(False, base_path)
    anno = load_annotated_corpus(False, base_path)

    click.secho(".done.", fg="green")
    click.secho(f"Number of News Documents (raw): {len(raw)}".rjust(50))
    click.secho(f"Number of News Documents (sentences): {len(sents)}".rjust(50))
    click.secho(f"Number of News Documents (annotated): {len(anno)}".rjust(50))

    if len(anno) != len(sents):
        anno_files = file_paths(CorpusTypeEnum.ANNOTATED, True, True, base_path)
        sent_files = file_paths(CorpusTypeEnum.SENTENCE, True, True, base_path)

        click.secho("\nSymmetric Difference between sentences & annotated corpus.")

        for diff in set(anno_files).symmetric_difference(set(sent_files)):
            click.secho(f"{diff}".rjust(50))
        click.secho(".warn", fg="yellow")

    click.secho("\nPerforming span checks...")

    for a, b, file in zip(raw, sents, file_paths()):
        for i, sent in enumerate(b['sentences']):
            if sent not in a:
                logger.error(f"""{sent}[{i}] \n\t\t is not a span in raw document \n {a} \n\n Corpus file: {file}
                """)
                sys.exit(1)

    click.secho(".done", fg="green")

    click.secho("\nPerforming span order checks...")

    for a, b, file in zip(raw, sents, file_paths()):

        start = 0
        for i, sent in enumerate(b['sentences']):

            idx = a.find(sent, start)

            if idx == -1:
                logger.error(
                    f"""{sent}[{i}] \n\t\t is potential our of order in "sentences" array of sentence corpus\n {a} \n\n Corpus file: {file}
                    """)
                sys.exit(1)
            else:
                start = start + len(sent)

    click.secho(".done", fg="green")

    click.secho("\nComparing annotated corpus with sentences corpus...")

    anno_names = file_paths(CorpusTypeEnum.ANNOTATED, noext=True, use_basename=True, base_path=base_path)
    sents_names = file_paths(CorpusTypeEnum.SENTENCE, noext=True, use_basename=True, base_path=base_path)

    anno_dict = dict((name, doc) for name, doc in zip(anno_names, anno))
    sents_dict = dict((name, doc) for name, doc in zip(sents_names, sents))

    match = 0

    for _name, _anno in anno_dict.items():
        sent = sents_dict[_name]

        if sent['sentences'] != _anno['sentences']:
            click.secho(f"\nSentences in annotated corpus {_name} doesn't match with document in sentence corpus.")
            sys.exit(1)
        else:
            match += 1

    click.secho(f".done ({match}/{len(anno_dict)})", fg="green")


if __name__ == '__main__':
    cli()
